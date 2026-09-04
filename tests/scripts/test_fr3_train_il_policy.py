#!/usr/bin/env python

from __future__ import annotations

import dataclasses
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


# ------------------------------------------------------------- dropping locked action dims ---


def test_action_drop_column_indices_finds_the_named_dims():
    names = ["delta.dx", "delta.dy", "delta.dz", "delta.drx", "delta.dry", "delta.drz", "gripper.pos"]

    assert fr3_train_il_policy.action_drop_column_indices(names, ["delta.drx", "delta.dry"]) == [3, 4]
    # Order of the request must not matter; the columns come out in column order.
    assert fr3_train_il_policy.action_drop_column_indices(names, ["delta.dry", "delta.drx"]) == [3, 4]


def test_action_drop_column_indices_refuses_a_name_that_is_not_an_action_dim():
    # A typo would otherwise drop nothing, and the only symptom would be a policy that keeps
    # predicting the axis the view was rebuilt to remove.
    names = ["delta.dx", "gripper.pos"]

    with pytest.raises(ValueError, match="not action dims"):
        fr3_train_il_policy.action_drop_column_indices(names, ["delta.drx"])


def test_action_drop_column_indices_refuses_to_drop_everything():
    names = ["delta.dx", "gripper.pos"]

    with pytest.raises(ValueError, match="every action dim"):
        fr3_train_il_policy.action_drop_column_indices(names, ["delta.dx", "gripper.pos"])


def test_view_leaves_out_the_dropped_action_dim(tmp_path):
    """The dropped dim must vanish from the column, the names, and the stats together.

    A view whose `names` still advertised the dim would be read back at deployment as an action
    the policy does not emit, and every dim after it would be off by one.
    """
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)

    frames, _episodes, info, manifest = _build_view(
        src_root, tmp_path / "view", action_drop_dims=["ee.x"]
    )

    assert info["features"]["action"]["names"] == ["gripper.pos"]
    assert info["features"]["action"]["shape"] == [1]
    assert all(len(row) == 1 for row in frames["action"])
    assert manifest["action_drop_dims"] == ["ee.x"]
    # The observation is untouched: only the action contract changed.
    assert info["features"]["observation.state"]["names"] == ["ee.x", "gripper.pos"]


def test_view_keeps_every_action_dim_when_nothing_is_dropped(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)

    _frames, _episodes, info, manifest = _build_view(src_root, tmp_path / "view")

    assert info["features"]["action"]["names"] == ["ee.x", "gripper.pos"]
    assert manifest["action_drop_dims"] == []


# --------------------------------------------------------- episode exclusion in the view ---


def _write_v3_source_dataset(
    root: Path,
    *,
    episodes: int = 3,
    frames: int = 4,
    camera: str = "observation.images.ee",
    fps: int = 30,
    task_prompt: str = "Pick and place",
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
        "fps": fps,
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

    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_prompt], name="task"))
    tasks.to_parquet(root / "meta" / "tasks.parquet")

    rows = {key: [] for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]}
    rows["observation.state"] = []
    rows["action"] = []
    for episode in range(episodes):
        for frame in range(frames):
            rows["timestamp"].append(frame / fps)
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
        "tasks": [[task_prompt]] * episodes,
        "length": [frames] * episodes,
        "data/chunk_index": [0] * episodes,
        "data/file_index": [0] * episodes,
        "dataset_from_index": [episode * frames for episode in range(episodes)],
        "dataset_to_index": [(episode + 1) * frames for episode in range(episodes)],
        f"videos/{camera}/chunk_index": [0] * episodes,
        f"videos/{camera}/file_index": [0] * episodes,
        f"videos/{camera}/from_timestamp": [episode * frames / fps for episode in range(episodes)],
        f"videos/{camera}/to_timestamp": [(episode + 1) * frames / fps for episode in range(episodes)],
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
    camera_crop_specs = kwargs.pop("camera_crop_specs", {})
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
        camera_crop_specs=camera_crop_specs,
        copy_videos=False,
        overwrite=True,
        **kwargs,
    )
    frames = pq.read_table(dst_root / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    episodes = pq.read_table(dst_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    info = json.loads((dst_root / "meta" / "info.json").read_text(encoding="utf-8"))
    manifest = json.loads((dst_root / "meta" / "il_view_manifest.json").read_text(encoding="utf-8"))
    return frames, episodes, info, manifest


def test_view_can_crop_camera_videos_without_mutating_recording(tmp_path, monkeypatch):
    src_root = tmp_path / "recording"
    dst_root = tmp_path / "view"
    _write_v3_source_dataset(src_root)
    calls: list[tuple[Path, Path, list[int]]] = []

    def fake_crop_video(src: Path, dst: Path, crop: list[int]) -> None:
        calls.append((src, dst, crop))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"cropped")

    monkeypatch.setattr(fr3_train_il_policy, "crop_video_file", fake_crop_video)

    _frames, _episodes, info, manifest = _build_view(
        src_root,
        dst_root,
        camera_crop_specs={"observation.images.ee": [2, 4, 20, 30]},
    )

    assert info["features"]["observation.images.ee"]["shape"] == [30, 20, 3]
    assert manifest["camera_crop_specs"] == {"observation.images.ee": [2, 4, 20, 30]}
    assert calls == [
        (
            src_root / "videos" / "observation.images.ee" / "chunk-000" / "file-000.mp4",
            dst_root / "videos" / "observation.images.ee" / "chunk-000" / "file-000.mp4",
            [2, 4, 20, 30],
        )
    ]
    assert calls[0][0].read_bytes() == b"\0" * 16
    assert calls[0][1].read_bytes() == b"cropped"


def test_camera_crop_validation_rejects_invalid_bounds():
    features = {"observation.images.ee": {"dtype": "video", "shape": [64, 64, 3]}}

    # The message names the frame the crop overran, because "which camera and how big is it"
    # is the whole question when a crop is rejected.
    with pytest.raises(ValueError, match=r"exceeds 64x64"):
        fr3_train_il_policy.validate_camera_crop_specs(
            {"observation.images.ee": [60, 0, 8, 8]}, features, ["observation.images.ee"]
        )

    with pytest.raises(ValueError, match="even"):
        fr3_train_il_policy.validate_camera_crop_specs(
            {"observation.images.ee": [1, 0, 8, 8]}, features, ["observation.images.ee"]
        )



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


# --------------------------------------------------------------- export frame rate ---


def test_view_decimates_a_faster_recording_to_the_requested_rate(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root, episodes=2, frames=6, fps=60)

    frames, episodes, info, manifest = _build_view(src_root, tmp_path / "view", view_fps=30)

    assert info["fps"] == 30
    assert info["total_frames"] == 6  # 2 episodes x ceil(6 / 2)
    assert list(episodes["length"]) == [3, 3]
    # Renumbered contiguously per episode, which is what frame_continuity checks.
    assert list(frames[frames["episode_index"] == 0]["frame_index"]) == [0, 1, 2]
    assert list(frames[frames["episode_index"] == 1]["frame_index"]) == [0, 1, 2]
    # Global row index stays dense across the whole view.
    assert list(frames["index"]) == list(range(6))
    # Kept rows are source frames 0, 2, 4 -- their observation marks them.
    assert [row[0] for row in frames[frames["episode_index"] == 0]["observation.state"]] == [0.0, 2.0, 4.0]
    # Timestamps are the source's own, and land on 1/30 spacing.
    kept = list(frames[frames["episode_index"] == 0]["timestamp"])
    assert kept == pytest.approx([0.0, 2 / 60, 4 / 60])
    assert kept == pytest.approx([n / 30 for n in range(3)])
    assert manifest["fps"] == 30
    assert list(manifest["frame_stride"].values()) == [2]
    assert list(manifest["source_fps"].values()) == [60]


def test_view_keeps_every_frame_when_the_rate_already_matches(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root, episodes=2, frames=4, fps=30)

    frames, _, info, manifest = _build_view(src_root, tmp_path / "view", view_fps=30)

    assert info["fps"] == 30
    assert info["total_frames"] == 8
    assert list(manifest["frame_stride"].values()) == [1]


def test_view_refuses_a_non_integer_frame_rate_ratio(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root, fps=60)

    with pytest.raises(ValueError, match="not an integer multiple"):
        _build_view(src_root, tmp_path / "view", view_fps=25)


def test_view_refuses_to_upsample(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root, fps=30)

    with pytest.raises(ValueError, match="below the requested"):
        _build_view(src_root, tmp_path / "view", view_fps=60)


def test_view_refuses_to_merge_disagreeing_rates_without_being_told_the_target(tmp_path):
    root = tmp_path / "sources"
    _write_v3_source_dataset(root / "slow", episodes=1, frames=4, fps=30)
    _write_v3_source_dataset(root / "fast", episodes=1, frames=4, fps=60)

    with pytest.raises(ValueError, match="disagree on fps"):
        _build_view(root, tmp_path / "view", view_fps=0)

    # With a target rate they merge: the point of the flag.
    _, _, info, manifest = _build_view(root, tmp_path / "merged", view_fps=30)
    assert info["fps"] == 30
    assert sorted(manifest["frame_stride"].values()) == [1, 2]


# ------------------------------------------------------------ training an existing view ---


def _view_with_manifest(tmp_path: Path, **overrides) -> Path:
    view_root = tmp_path / "view"
    (view_root / "meta").mkdir(parents=True)
    manifest = {
        "repo_id": "local/fr3__delta_ee_from_prev_cmd",
        "cameras": ["observation.images.ee", "observation.images.side"],
        "state_keys": ["observation.state"],
        "action_append_selectors": [],
        "action_append_names": [],
        "image_resize_shape": None,
        "action_mode": "delta_ee_from_prev_cmd",
        "fps": 30,
        "total_episodes": 20,
        "total_rows": 10305,
    }
    manifest.update(overrides)
    (view_root / "meta" / "il_view_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return view_root


def _args(**overrides):
    parser = fr3_train_il_policy.build_arg_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_adopting_a_view_takes_its_shape_from_the_manifest(tmp_path):
    """The frames already exist; the CLI's idea of what is in them is not evidence.

    A --cameras that disagreed with the view would not fail here -- it would fail deep in
    training, or produce a checkpoint expecting a camera the rollout never sends.
    """
    view_root = _view_with_manifest(tmp_path)
    args = _args(cameras="observation.images.wrong", state_keys="observation.nonsense")

    manifest = fr3_train_il_policy.adopt_existing_view(args, view_root)

    assert args.cameras == "observation.images.ee,observation.images.side"
    assert args.state_keys == "observation.state"
    assert args.action_mode == "delta_ee_from_prev_cmd"
    assert args.repo_id == "local/fr3__delta_ee_from_prev_cmd"
    assert manifest["total_episodes"] == 20


def test_adopting_a_view_inherits_its_resize_but_lets_the_caller_override(tmp_path):
    view_root = _view_with_manifest(tmp_path, image_resize_shape=[240, 320])

    inherited = _args(image_resize_shape=None)
    fr3_train_il_policy.adopt_existing_view(inherited, view_root)
    assert inherited.image_resize_shape == "240,320"

    # A resize is a training-time transform, so asking for a different one is legitimate.
    overridden = _args(image_resize_shape="120,160")
    fr3_train_il_policy.adopt_existing_view(overridden, view_root)
    assert overridden.image_resize_shape == "120,160"


def test_adopting_a_missing_view_says_where_it_looked(tmp_path):
    with pytest.raises(ValueError, match="il_view_manifest.json"):
        fr3_train_il_policy.adopt_existing_view(_args(), tmp_path / "never-built")


def test_training_a_shared_view_keeps_each_jobs_configs_apart(tmp_path, monkeypatch):
    """Two jobs can train one view; neither may leave its settings in the other's way.

    The view is built once and trained repeatedly -- different policies, different step
    counts. Writing the generated configs to the view root would mean the inference config
    there names whichever job ran last, which is exactly the kind of quietly-wrong pointer
    the record-config derivation was introduced to remove.

    They also must not go in the training output directory: lerobot_train refuses to start
    when that already exists and it is not resuming, so a config placed there would make
    every fresh run fail on scaffolding this script had just created.
    """
    view_root = _view_with_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fr3_train_il_policy.py",
            "--skip-prepare",
            "--view-root", str(view_root),
            "--job-name", "baseline__act",
            "--output-dir", str(tmp_path / "train" / "baseline__act"),
            "--policy", "act",
            "--prepare-only",
        ],
    )
    # --prepare-only is refused together with --skip-prepare, which is what this asserts first.
    with pytest.raises(ValueError, match="opposite halves"):
        fr3_train_il_policy.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "fr3_train_il_policy.py",
            "--skip-prepare",
            "--view-root", str(view_root),
            "--job-name", "baseline__act",
            "--output-dir", str(tmp_path / "train" / "baseline__act"),
            "--policy", "act",
            "--steps", "1",
        ],
    )
    monkeypatch.setattr(fr3_train_il_policy.subprocess, "run", lambda *a, **k: None)
    fr3_train_il_policy.main()

    run_dir = view_root / "runs" / "baseline__act"
    assert (run_dir / "train_config.generated.json").is_file()
    assert (run_dir / "inference_config.generated.yaml").is_file()
    # The view root is left exactly as the export step wrote it.
    assert not (view_root / "train_config.generated.json").exists()
    assert not (view_root / "inference_config.generated.yaml").exists()
    # And the output directory is untouched, so lerobot_train can create it itself.
    assert not (tmp_path / "train" / "baseline__act").exists()

    config = json.loads((run_dir / "train_config.generated.json").read_text(encoding="utf-8"))
    assert config["output_dir"] == str(tmp_path / "train" / "baseline__act")
    assert config["dataset"]["root"] == str(view_root)
    assert config["dataset"]["repo_id"] == "local/fr3__delta_ee_from_prev_cmd"
    assert config["policy"]["type"] == "act"
    assert config["job_name"] == "baseline__act"


# --------------------------------------------------------------------------------------
# Task prompt (the language a VLA is conditioned on)
# --------------------------------------------------------------------------------------


def _view_tasks(dst_root: Path) -> list[str]:
    """The view's prompts in task_index order -- the table a sample's `task` is resolved from."""
    return fr3_train_il_policy.view_task_prompts(dst_root)


def _task_indices(dst_root: Path, data_file: str) -> set[int]:
    """The task indices in one of a merged view's data files."""
    import pyarrow.parquet as pq

    table = pq.read_table(dst_root / "data" / "chunk-000" / f"{data_file}.parquet")
    return set(table.column("task_index").to_pylist())


def test_a_view_keeps_the_recorded_prompt_when_none_is_given(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root, task_prompt="Pick and place")

    _frames, episodes, _info, manifest = _build_view(src_root, tmp_path / "view")

    assert _view_tasks(tmp_path / "view") == ["Pick and place"]
    assert list(episodes["tasks"].iloc[0]) == ["Pick and place"]
    assert manifest["task_prompt_override"] is None
    assert manifest["task_prompts"] == ["Pick and place"]


def test_a_view_can_be_given_a_better_prompt_than_the_recording_carried(tmp_path):
    """The prompt is tokenized into every pi0/pi0.5 sample, so it is data, not a run setting."""
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root, task_prompt="Pick and place")

    frames, episodes, info, manifest = _build_view(
        src_root,
        tmp_path / "view",
        task_prompt="pick up the red cube and place it in the box",
    )

    assert _view_tasks(tmp_path / "view") == ["pick up the red cube and place it in the box"]
    # Every frame still points at the one task row, and the human-readable copy agrees with it.
    assert set(frames["task_index"]) == {0}
    assert list(episodes["tasks"].iloc[0]) == ["pick up the red cube and place it in the box"]
    assert info["total_tasks"] == 1
    assert manifest["task_prompt_override"] == "pick up the red cube and place it in the box"
    assert manifest["source_task_prompts"] == ["Pick and place"]

    # The recording is untouched: it is the only primary record of what was captured.
    source_tasks = pd.read_parquet(src_root / "meta" / "tasks.parquet")
    assert list(source_tasks.index) == ["Pick and place"]


def test_a_prompt_is_normalized_the_way_the_tokenizer_will_see_it(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)

    _build_view(src_root, tmp_path / "view", task_prompt="  put  the cube\n in the box ")

    # Not cosmetic: pi0.5 builds `Task: {task}, State: ...` and tokenizes it, so a stray double
    # space is a different token sequence conditioning every frame in the dataset.
    assert _view_tasks(tmp_path / "view") == ["put the cube in the box"]


def test_a_merge_can_rewrite_each_recordings_prompt_separately(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_v3_source_dataset(first, episodes=2, task_prompt="Pick and place")
    _write_v3_source_dataset(second, episodes=2, task_prompt="grab thing")

    _frames, episodes, info, manifest = _build_view(
        [first, second],
        tmp_path / "view",
        task_prompt_map={
            "Pick and place": "pick up the red cube and place it in the box",
            "grab thing": "pick up the blue cube and place it in the box",
        },
    )

    assert _view_tasks(tmp_path / "view") == [
        "pick up the red cube and place it in the box",
        "pick up the blue cube and place it in the box",
    ]
    assert info["total_tasks"] == 2
    # Each source keeps its own file in a merge, and each source's frames follow its own rewrite
    # through the renumbered index space.
    assert _task_indices(tmp_path / "view", "file-000") == {0}
    assert _task_indices(tmp_path / "view", "file-001") == {1}
    assert list(episodes["tasks"].iloc[0]) == ["pick up the red cube and place it in the box"]
    assert list(episodes["tasks"].iloc[-1]) == ["pick up the blue cube and place it in the box"]
    assert manifest["source_task_prompts"] == ["Pick and place", "grab thing"]


def test_two_recordings_can_be_collapsed_onto_one_instruction(tmp_path):
    """Two prompts rewritten to one string *are* one task: the index space is rebuilt here."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_v3_source_dataset(first, episodes=2, task_prompt="Pick and place")
    _write_v3_source_dataset(second, episodes=2, task_prompt="pick+place v2")

    _frames, _episodes, info, _manifest = _build_view(
        [first, second], tmp_path / "view", task_prompt="pick up the cube and place it in the box"
    )

    assert _view_tasks(tmp_path / "view") == ["pick up the cube and place it in the box"]
    assert info["total_tasks"] == 1
    assert _task_indices(tmp_path / "view", "file-000") == {0}
    assert _task_indices(tmp_path / "view", "file-001") == {0}


def test_a_prompt_map_refuses_a_prompt_no_recording_has(tmp_path):
    """Silently ignoring it would train the old wording while the command line said otherwise."""
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root, task_prompt="Pick and place")
    dst_root = tmp_path / "view"

    with pytest.raises(ValueError, match="no source dataset records"):
        _build_view(src_root, dst_root, task_prompt_map={"Pick and Place": "pick up the red cube"})

    # And nothing was written. A rejected build that left the directory behind would make the
    # retry -- the one with the typo fixed -- fail on "already exists" instead of running.
    assert not dst_root.exists()


def test_the_two_prompt_flags_cannot_both_rewrite_the_same_column(tmp_path):
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)

    with pytest.raises(ValueError, match="both rewrite"):
        _build_view(
            src_root,
            tmp_path / "view",
            task_prompt="one thing",
            task_prompt_map={"Pick and place": "another thing"},
        )


def test_the_digest_changes_when_only_the_prompt_does(tmp_path):
    """Two views over the same frames with different instructions are not the same training set."""
    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)

    _f, _e, _i, recorded = _build_view(src_root, tmp_path / "as-recorded")
    _f, _e, _i, rewritten = _build_view(src_root, tmp_path / "rewritten", task_prompt="do the thing")

    assert recorded["source_digest"] != rewritten["source_digest"]


def test_a_prompt_rewrite_is_refused_when_no_view_is_being_built(tmp_path):
    """--skip-prepare trains frames whose task column is already on disk."""
    view_root = _view_with_manifest(tmp_path)
    args = _args(skip_prepare=True, task_prompt="do the thing")

    with pytest.raises(ValueError, match="Rebuild the view"):
        fr3_train_il_policy.validate_prompt_args(args)

    resumed = _args(resume=True, task_prompt="do the thing")
    with pytest.raises(ValueError, match="two different instructions"):
        fr3_train_il_policy.validate_prompt_args(resumed)

    # And is a no-op to validate when nothing was asked for.
    fr3_train_il_policy.validate_prompt_args(_args(skip_prepare=True))
    assert view_root.exists()


def test_the_generated_inference_config_carries_the_prompt_to_the_rollout(tmp_path, monkeypatch):
    """pi0.5 takes the task from the caller at inference, not from the checkpoint."""
    import yaml

    src_root = tmp_path / "recording"
    _write_v3_source_dataset(src_root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fr3_train_il_policy.py",
            "--dataset-roots", str(src_root),
            "--view-root", str(tmp_path / "view"),
            "--job-name", "cube__pi05",
            "--cameras", "observation.images.ee",
            "--state-keys", "observation.state",
            "--action-append-selectors", "",
            "--action-append-names", "",
            "--policy", "pi05",
            "--task-prompt", "pick up the red cube and place it in the box",
            "--prepare-only",
        ],
    )
    fr3_train_il_policy.main()

    config = yaml.safe_load(
        (tmp_path / "view" / "inference_config.generated.yaml").read_text(encoding="utf-8")
    )
    assert config["runtime"]["task_prompt"] == "pick up the red cube and place it in the box"
    assert config["training"]["task_prompts"] == ["pick up the red cube and place it in the box"]


# --------------------------------------------------------------------------------------
# Finetuning from a base checkpoint, with and without LoRA
# --------------------------------------------------------------------------------------


def test_a_base_checkpoint_supplies_weights_and_nothing_else(tmp_path):
    policy = fr3_train_il_policy.build_policy_section(
        _args(policy="pi05", pretrained_path="lerobot/pi05_base"), None
    )

    assert policy["pretrained_path"] == "lerobot/pi05_base"
    # `use_peft` means "the path points at an adapter". Setting it for a LoRA run would send
    # make_policy looking for an adapter config inside a base model.
    assert "use_peft" not in policy


def test_a_dense_run_emits_no_peft_block():
    assert fr3_train_il_policy.build_peft_section(_args(policy="pi05")) is None


def test_lora_emits_the_block_lerobot_train_reads():
    peft = fr3_train_il_policy.build_peft_section(
        _args(policy="pi05", lora=True, pretrained_path="lerobot/pi05_base", lora_r=32)
    )

    assert peft == {"method_type": "LORA", "r": 32, "alpha": 32}
    # Every key here has to be a field of lerobot.configs.default.PeftConfig -- draccus parses
    # this block into that dataclass, and an unknown key is a parse error, not an ignored one.
    from lerobot.configs.default import PeftConfig

    assert set(peft) <= {field.name for field in dataclasses.fields(PeftConfig)}


def test_alpha_tracks_the_rank_so_a_higher_rank_is_a_stronger_adapter():
    """An adapter's strength is alpha/r, and PEFT's own alpha default is a fixed 8.

    Left to that default, raising --lora-r *weakens* every update (r=32 -> 0.25) instead of
    strengthening it, which is the opposite of what the flag advertises and silently
    attenuated every capacity experiment on this rig.
    """
    for rank in (8, 16, 32, 64):
        peft = fr3_train_il_policy.build_peft_section(
            _args(policy="pi05", lora=True, pretrained_path="lerobot/pi05_base", lora_r=rank)
        )
        assert peft["alpha"] / peft["r"] == 1.0


def test_an_explicit_alpha_wins_over_the_rank():
    """Passing 8 has to reproduce PEFT's old fixed default exactly, for comparability with
    checkpoints trained before alpha was reachable."""
    peft = fr3_train_il_policy.build_peft_section(
        _args(
            policy="pi05",
            lora=True,
            pretrained_path="lerobot/pi05_base",
            lora_r=32,
            lora_alpha=8,
        )
    )

    assert (peft["r"], peft["alpha"]) == (32, 8)


def test_lora_leaves_the_targets_alone_unless_asked():
    """pi0.5's own default target set is the tuned one; a null would overwrite it."""
    default = fr3_train_il_policy.build_peft_section(
        _args(policy="pi05", lora=True, pretrained_path="lerobot/pi05_base")
    )
    assert "target_modules" not in default
    assert "full_training_modules" not in default

    # A single token stays one string: 'all-linear' is a PEFT keyword and a regex is not a list.
    keyword = fr3_train_il_policy.build_peft_section(
        _args(
            policy="pi05",
            lora=True,
            pretrained_path="lerobot/pi05_base",
            lora_target_modules="all-linear",
        )
    )
    assert keyword["target_modules"] == "all-linear"

    listed = fr3_train_il_policy.build_peft_section(
        _args(
            policy="pi05",
            lora=True,
            pretrained_path="lerobot/pi05_base",
            lora_target_modules="q_proj,v_proj",
            lora_full_training_modules="",
        )
    )
    assert listed["target_modules"] == ["q_proj", "v_proj"]
    # An explicit empty string means "none", which is not the same as not asking.
    assert listed["full_training_modules"] == []


def test_lora_without_a_base_model_is_refused_before_the_run_starts():
    """The same refusal PEFT makes, but before a dataset scan and a policy build."""
    with pytest.raises(ValueError, match="nothing here to adapt"):
        fr3_train_il_policy.validate_finetune_args(_args(policy="pi05", lora=True))

    fr3_train_il_policy.validate_finetune_args(
        _args(policy="pi05", lora=True, pretrained_path="lerobot/pi05_base")
    )


def test_a_pi05_lora_run_writes_a_config_lerobot_train_can_parse(tmp_path, monkeypatch):
    view_root = _view_with_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fr3_train_il_policy.py",
            "--skip-prepare",
            "--view-root", str(view_root),
            "--job-name", "cube__pi05_lora",
            "--output-dir", str(tmp_path / "train" / "cube__pi05_lora"),
            "--policy", "pi05",
            "--pretrained-path", "lerobot/pi05_base",
            "--lora",
            "--lora-r", "32",
            "--steps", "1",
        ],
    )
    monkeypatch.setattr(fr3_train_il_policy.subprocess, "run", lambda *a, **k: None)
    fr3_train_il_policy.main()

    config = json.loads(
        (view_root / "runs" / "cube__pi05_lora" / "train_config.generated.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["policy"]["type"] == "pi05"
    assert config["policy"]["pretrained_path"] == "lerobot/pi05_base"
    # Top level, not under `policy`: TrainPipelineConfig.peft is what lerobot_train checks
    # before calling `policy.wrap_with_peft`.
    assert config["peft"] == {"method_type": "LORA", "r": 32, "alpha": 32}

    from lerobot.configs.train import TrainPipelineConfig

    assert "peft" in {field.name for field in dataclasses.fields(TrainPipelineConfig)}


# --------------------------------------------------------------------------------------
# Explicit multi-source selection
# --------------------------------------------------------------------------------------


def test_an_explicit_root_list_is_taken_literally(tmp_path):
    """A ticked list must not be expanded into the directory that holds it.

    ``discover_dataset_roots`` treats a *directory* as "everything inside", which is what the
    GUI's selection has to be protected from: the datasets root holds every recording on the
    machine, and pulling in the unticked ones would make the training set differ from what the
    page said it was building.
    """
    first = tmp_path / "datasets" / "pick_and_place_20260819_171323"
    second = tmp_path / "datasets" / "pick_and_place_20260819_171756"
    unselected = tmp_path / "datasets" / "stack_cube_20260819_180000"
    for root in (first, second, unselected):
        _write_v3_source_dataset(root, episodes=1)

    assert fr3_train_il_policy.discover_dataset_roots([first, second]) == [first, second]
    # The same parent, passed as a directory, is the "everything inside" form.
    assert unselected in fr3_train_il_policy.discover_dataset_roots(tmp_path / "datasets")


def test_an_explicit_list_rejects_an_entry_that_is_not_a_dataset(tmp_path):
    good = tmp_path / "good"
    _write_v3_source_dataset(good, episodes=1)
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="not a LeRobot dataset root"):
        fr3_train_il_policy.discover_dataset_roots([good, empty])


def test_an_explicit_list_drops_a_root_named_twice(tmp_path):
    root = tmp_path / "recording"
    _write_v3_source_dataset(root, episodes=1)

    # Duplicated selection would otherwise double every episode, and the view has no way to
    # say that two of its episodes are the same frames.
    assert fr3_train_il_policy.discover_dataset_roots([root, root]) == [root.resolve()]


def test_a_merge_records_each_source_and_a_digest_of_the_selection(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_v3_source_dataset(first, episodes=2)
    _write_v3_source_dataset(second, episodes=1)

    _frames, episodes, info, manifest = _build_view([first, second], tmp_path / "view")

    assert manifest["source_dataset_roots"] == [str(first), str(second)]
    # No single source: a merged view that named one of them would misdescribe itself.
    assert manifest["source_dataset_root"] is None
    assert info["total_episodes"] == 3
    assert len(episodes) == 3
    assert manifest["build_id"]
    # Episodes are renumbered, so this is the only way back to where each one came from.
    assert {entry["source_dataset_root"] for entry in manifest["episode_source_index"]} == {
        str(first),
        str(second),
    }


def test_the_source_digest_changes_when_the_selection_does(tmp_path):
    """Views are rebuilt under the same name, so the path alone no longer identifies the frames.

    A checkpoint that records the digest can still tell whether the view on disk is the one it
    trained on -- without it, adding a session to a task silently redefines every checkpoint's
    training set.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_v3_source_dataset(first, episodes=2)
    _write_v3_source_dataset(second, episodes=1)

    _, _, _, one = _build_view([first], tmp_path / "one")
    _, _, _, again = _build_view([first], tmp_path / "again")
    _, _, _, both = _build_view([first, second], tmp_path / "both")

    assert one["source_digest"] == again["source_digest"]
    assert one["source_digest"] != both["source_digest"]


def _write_resume_checkpoint(
    root: Path, *, dataset_root: Path, step: int = 30000, episodes: list[int] | None = None
) -> Path:
    """A checkpoint directory shaped the way lerobot_train saves one."""
    checkpoint = root / "checkpoints" / f"{step:06d}"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=True)
    dataset: dict = {"repo_id": f"local/{dataset_root.name}", "root": str(dataset_root)}
    if episodes is not None:
        dataset["episodes"] = episodes
    (pretrained / "train_config.json").write_text(
        json.dumps({"dataset": dataset, "steps": step, "job_name": root.name}), encoding="utf-8"
    )
    state = checkpoint / "training_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "training_step.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    return pretrained / "train_config.json"


def test_resume_refuses_a_checkpoint_trained_on_a_different_view(tmp_path):
    """The failure this replaces is silent: `--view-root` is printed and then ignored.

    `TrainPipelineConfig.validate` reloads the whole config from the checkpoint when resuming, so
    a run launched against a merged DAgger view trains the old base view, writes into the old
    run's output directory, and exits reporting success.
    """
    old_view = tmp_path / "views" / "base_only"
    new_view = tmp_path / "views" / "base_plus_dagger"
    old_view.mkdir(parents=True)
    new_view.mkdir(parents=True)
    config_path = _write_resume_checkpoint(tmp_path / "train" / "L4", dataset_root=old_view)

    with pytest.raises(ValueError) as excinfo:
        fr3_train_il_policy.assert_resume_trains_the_named_view(
            config_path,
            view_root=new_view,
            repo_id=f"local/{new_view.name}",
            steps=30000,
            steps_supplied=True,
        )
    assert str(old_view) in str(excinfo.value)
    assert "--pretrained-path" in str(excinfo.value)


def test_resume_refuses_a_run_that_would_take_zero_steps(tmp_path):
    """Resuming a finished checkpoint at its own step count exits without training anything."""
    view = tmp_path / "views" / "one"
    view.mkdir(parents=True)
    config_path = _write_resume_checkpoint(tmp_path / "train" / "L4", dataset_root=view, step=30000)

    with pytest.raises(ValueError, match="zero steps"):
        fr3_train_il_policy.assert_resume_trains_the_named_view(
            config_path, view_root=view, repo_id="local/one", steps=30000, steps_supplied=True
        )


def test_resume_allows_continuing_the_same_view_past_its_saved_step(tmp_path):
    """The one thing resume is for still works, episode subset and all."""
    view = tmp_path / "views" / "one"
    view.mkdir(parents=True)
    config_path = _write_resume_checkpoint(
        tmp_path / "train" / "L4", dataset_root=view, step=30000, episodes=[0, 1, 2]
    )

    fr3_train_il_policy.assert_resume_trains_the_named_view(
        config_path, view_root=view, repo_id="local/one", steps=40000, steps_supplied=True
    )
