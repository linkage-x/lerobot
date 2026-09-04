from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.fr3.fr3_merge_policy_ready_datasets import (
    IS_INTERVENTION_KEY,
    MergeError,
    merge_policy_ready_datasets,
    validate_policy_ready_merge,
)


def _features(*, include_intervention: bool = False) -> dict:
    features = {
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "observation.state": {
            "dtype": "float32",
            "shape": [3],
            "names": ["ee.x", "ee.y", "gripper.pos"],
        },
        "action": {
            "dtype": "float32",
            "shape": [3],
            "names": ["delta_ee_from_prev_cmd.dx", "delta_ee_from_prev_cmd.dy", "gripper.pos"],
        },
        "observation.images.ee": {"dtype": "video", "shape": [64, 64, 3]},
    }
    if include_intervention:
        features[IS_INTERVENTION_KEY] = {"dtype": "float32", "shape": [1], "names": [IS_INTERVENTION_KEY]}
    return features


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_dataset(
    root: Path,
    *,
    episodes: int,
    frames_per_episode: int = 2,
    prompt: str = "pick and place the cube",
    include_intervention: bool = False,
    intervention_value: float = 1.0,
    intervention_as_list: bool = False,
    fps: int = 30,
    action_names: list[str] | None = None,
) -> None:
    features = _features(include_intervention=include_intervention)
    if action_names is not None:
        features["action"]["names"] = action_names
    total_frames = episodes * frames_per_episode
    _write_json(
        root / "meta" / "info.json",
        {
            "repo_id": f"local/{root.name}",
            "fps": fps,
            "total_episodes": episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "splits": {"train": f"0:{episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        },
    )
    _write_json(
        root / "meta" / "il_view_manifest.json",
        {
            "repo_id": f"local/{root.name}",
            "action_mode": "delta_ee_from_prev_cmd",
            "task_prompts": [prompt],
            "fps": fps,
            "total_episodes": episodes,
            "total_rows": total_frames,
        },
    )
    pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([prompt], name="task"),
    ).to_parquet(root / "meta" / "tasks.parquet")

    rows = {
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
        "observation.state": [],
        "action": [],
    }
    if include_intervention:
        rows[IS_INTERVENTION_KEY] = []
    index = 0
    for episode in range(episodes):
        for frame in range(frames_per_episode):
            rows["timestamp"].append(frame / fps)
            rows["frame_index"].append(frame)
            rows["episode_index"].append(episode)
            rows["index"].append(index)
            rows["task_index"].append(0)
            rows["observation.state"].append(np.array([episode, frame, 0.5], dtype=np.float32))
            rows["action"].append(np.array([0.01 * (frame + 1), 0.0, 0.5], dtype=np.float32))
            if include_intervention:
                # Scalar by default, because that is what the recorder's own writer produces for
                # a shape-[1] feature. `intervention_as_list` reproduces the length-1 list a
                # hand-written producer can leave behind, which the merge has to flatten.
                rows[IS_INTERVENTION_KEY].append(
                    np.array([intervention_value], dtype=np.float32)
                    if intervention_as_list
                    else np.float32(intervention_value)
                )
            index += 1
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), data_dir / "file-000.parquet")

    episodes_rows = []
    for episode in range(episodes):
        start = episode * frames_per_episode
        episodes_rows.append(
            {
                "episode_index": episode,
                "tasks": [prompt],
                "length": frames_per_episode,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": start,
                "dataset_to_index": start + frames_per_episode,
                "videos/observation.images.ee/chunk_index": 0,
                "videos/observation.images.ee/file_index": 0,
                "videos/observation.images.ee/from_timestamp": 0.0,
                "videos/observation.images.ee/to_timestamp": frames_per_episode / fps,
            }
        )
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(episodes_rows), preserve_index=False), ep_path)

    video = root / "videos" / "observation.images.ee" / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video" * 512)

    # The full vocabulary a LeRobot writer records, not just the moments: a policy's
    # `normalization_mapping` decides which of these it reads, and pi0.5 asks STATE for
    # QUANTILES. A fixture that omitted them could not notice a merge dropping them.
    def _stats(dim: int) -> dict:
        return {
            "count": [total_frames],
            **{
                key: [float(index) for index in range(dim)]
                for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
            },
        }

    _write_json(
        root / "meta" / "stats.json",
        {
            "observation.state": _stats(3),
            "action": _stats(3),
            "observation.images.ee": _stats(3),
        },
    )


def _write_qc(root: Path, status: str = "pass") -> None:
    _write_json(
        root / "meta" / "processing.json",
        {
            "active_version": "v1",
            "versions": {
                "v1": {
                    "qc": {
                        "status": status,
                        "summary": f"qc {status}",
                        "valid_frames_pct": 100.0 if status == "pass" else 0.0,
                        "checks": [{"name": "schema", "status": status if status != "failed" else "fail", "message": "ok"}],
                    }
                }
            },
        },
    )


def test_policy_ready_merge_checks_and_merges_base_view_with_dagger(tmp_path):
    base = tmp_path / "L4_full48_holdout22_40"
    dagger = tmp_path / "dagger_L4_full48_holdout22_40_030000"
    out = tmp_path / "L4_full48_holdout22_40_plus_dagger_030000"
    _write_dataset(base, episodes=2, include_intervention=False)
    _write_dataset(dagger, episodes=1, include_intervention=True, intervention_value=1.0)
    _write_qc(dagger, "pass")

    check = validate_policy_ready_merge(base, [dagger])
    assert check["ok"] is True
    assert check["totalEpisodes"] == 3

    result = merge_policy_ready_datasets(
        base_view=base,
        dagger_roots=[dagger],
        output_root=out,
        repo_id="local/combined",
    )

    assert result["ok"] is True
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 6
    assert IS_INTERVENTION_KEY in info["features"]
    data = pd.concat(
        [pq.read_table(path).to_pandas() for path in sorted((out / "data").glob("chunk-*/*.parquet"))],
        ignore_index=True,
    )
    assert data["episode_index"].tolist() == [0, 0, 1, 1, 2, 2]
    assert data["index"].tolist() == list(range(6))
    assert [float(value) for value in data[IS_INTERVENTION_KEY].tolist()] == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    manifest = json.loads((out / "meta" / "il_view_manifest.json").read_text(encoding="utf-8"))
    assert manifest["merge_type"] == "policy_ready_dagger"
    assert manifest["base_training_view_root"] == str(base)
    assert manifest["dagger_dataset_roots"] == [str(dagger)]




def test_policy_ready_merge_writes_is_intervention_as_the_scalar_lerobot_reads(tmp_path):
    """Every data file must agree, and agree on a bare float.

    `get_hf_features_from_features` maps a shape-[1] feature to `datasets.Value`, so a length-1
    list here raises `Couldn't cast array of type list<element: float> to float` when training
    opens the dataset. Nothing before that point notices: meta/info.json carries the same feature
    dict either way, which is why this is asserted against the parquet rather than the metadata.
    """
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    out = tmp_path / "combined"
    _write_dataset(base, episodes=2, include_intervention=False)
    _write_dataset(dagger, episodes=1, include_intervention=True)
    _write_qc(dagger, "pass")

    merge_policy_ready_datasets(
        base_view=base, dagger_roots=[dagger], output_root=out, repo_id="local/combined"
    )

    types = {
        path.name: str(pq.read_schema(path).field(IS_INTERVENTION_KEY).type)
        for path in sorted((out / "data").glob("chunk-*/*.parquet"))
    }
    assert set(types.values()) == {"float"}, types


def test_policy_ready_merge_flattens_a_list_spelled_intervention_column(tmp_path):
    """A source that wrote the flag as a length-1 list must not carry that into the output."""
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    out = tmp_path / "combined"
    _write_dataset(base, episodes=2, include_intervention=False)
    _write_dataset(dagger, episodes=1, include_intervention=True, intervention_as_list=True)
    _write_qc(dagger, "pass")

    merge_policy_ready_datasets(
        base_view=base, dagger_roots=[dagger], output_root=out, repo_id="local/combined"
    )

    data = pd.concat(
        [pq.read_table(path).to_pandas() for path in sorted((out / "data").glob("chunk-*/*.parquet"))],
        ignore_index=True,
    )
    assert [float(value) for value in data[IS_INTERVENTION_KEY].tolist()] == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    types = {str(pq.read_schema(path).field(IS_INTERVENTION_KEY).type)
             for path in sorted((out / "data").glob("chunk-*/*.parquet"))}
    assert types == {"float"}


def test_policy_ready_merge_keeps_the_quantiles_a_policy_normalizes_on(tmp_path):
    """pi0.5 normalizes STATE with QUANTILES, so q01/q99 are load-bearing, not decoration.

    Without them the dataset merges, passes every check, opens fine, and then raises
    "QUANTILES normalization mode requires q01 and q99 stats" inside the first training step --
    after the base model has been loaded onto the GPU.
    """
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    out = tmp_path / "combined"
    _write_dataset(base, episodes=2, include_intervention=False)
    _write_dataset(dagger, episodes=1, include_intervention=True)
    _write_qc(dagger, "pass")

    merge_policy_ready_datasets(
        base_view=base, dagger_roots=[dagger], output_root=out, repo_id="local/combined"
    )

    stats = json.loads((out / "meta" / "stats.json").read_text(encoding="utf-8"))
    for key in ("observation.state", "action"):
        assert {"count", "min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"} <= set(
            stats[key]
        ), f"{key} lost statistics: {sorted(stats[key])}"
        assert stats[key]["count"] == [6]
        # Recomputed over the merged rows, so the quantiles have to bracket the data rather than
        # be inherited from a base view whose episodes are only partly present.
        assert all(
            low <= high for low, high in zip(stats[key]["q01"], stats[key]["q99"], strict=True)
        )


def test_policy_ready_merge_can_keep_a_base_episode_subset(tmp_path):
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    out = tmp_path / "combined"
    _write_dataset(base, episodes=3, include_intervention=False)
    _write_dataset(dagger, episodes=1, include_intervention=True, intervention_value=1.0)
    _write_qc(dagger, "pass")

    check = validate_policy_ready_merge(base, [dagger], base_episodes=[0, 2])
    assert check["totalEpisodes"] == 3
    assert check["totalFrames"] == 6

    result = merge_policy_ready_datasets(
        base_view=base,
        dagger_roots=[dagger],
        output_root=out,
        repo_id="local/combined",
        base_episodes=[0, 2],
    )

    assert result["totalEpisodes"] == 3
    data = pd.concat(
        [pq.read_table(path).to_pandas() for path in sorted((out / "data").glob("chunk-*/*.parquet"))],
        ignore_index=True,
    )
    assert data["episode_index"].tolist() == [0, 0, 1, 1, 2, 2]
    assert [np.asarray(value).tolist() for value in data["observation.state"]] == [
        [0.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [2.0, 0.0, 0.5],
        [2.0, 1.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
    ]
    manifest = json.loads((out / "meta" / "il_view_manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_episode_filter"] == [0, 2]
    assert manifest["episode_source_index"][:2] == [
        {"episode_index": 0, "source_dataset_root": str(base), "source_episode_index": 0},
        {"episode_index": 1, "source_dataset_root": str(base), "source_episode_index": 2},
    ]

def test_policy_ready_merge_requires_dagger_qc_pass(tmp_path):
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    _write_dataset(base, episodes=1)
    _write_dataset(dagger, episodes=1, include_intervention=True)
    _write_qc(dagger, "fail")

    with pytest.raises(MergeError, match="must be QC PASS"):
        validate_policy_ready_merge(base, [dagger])


def test_policy_ready_merge_refuses_schema_mismatch(tmp_path):
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    _write_dataset(base, episodes=1)
    _write_dataset(
        dagger,
        episodes=1,
        include_intervention=True,
        action_names=["wrong.dx", "wrong.dy", "gripper.pos"],
    )
    _write_qc(dagger, "pass")

    with pytest.raises(MergeError, match="schema differs"):
        validate_policy_ready_merge(base, [dagger])


def test_policy_ready_merge_refuses_prompt_mismatch(tmp_path):
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    _write_dataset(base, episodes=1, prompt="pick cube")
    _write_dataset(dagger, episodes=1, prompt="stack cube", include_intervention=True)
    _write_qc(dagger, "pass")

    with pytest.raises(MergeError, match="prompt set"):
        validate_policy_ready_merge(base, [dagger])
