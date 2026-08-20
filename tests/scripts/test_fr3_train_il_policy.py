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
    root: Path,
    *,
    episodes: int = 3,
    frames: int = 4,
    camera: str = "observation.images.ee",
    fps: int = 30,
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

    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index(["Pick and place"], name="task"))
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
        "tasks": [["Pick and place"]] * episodes,
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
