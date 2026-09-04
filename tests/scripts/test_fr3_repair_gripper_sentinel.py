from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.fr3.fr3_repair_gripper_sentinel import (
    BRACKET_TOLERANCE,
    MIN_BRACKETED_RUNS,
    REPAIR_DIRNAME,
    SENTINEL_ISOLATION,
    STATS_KEYS,
    Audit,
    RepairError,
    apply_repair,
    audit_recording,
    episode_bounds,
    main,
    patch_stats_vector,
    premise_failures,
    read_provenance,
    read_recording,
    repair_values,
    sentinel_runs,
    stale_views,
    verify_written,
)

STATE_NAMES = ["ee.x", "gripper.pos", "prev_cmd.gripper.pos"]
GRIPPER = STATE_NAMES.index("gripper.pos")


def _bounds(*lengths: int) -> list[tuple[int, int, int]]:
    bounds, start = [], 0
    for episode, length in enumerate(lengths):
        bounds.append((episode, start, start + length))
        start += length
    return bounds


# --------------------------------------------------------------------------------------
# Finding the sentinel runs
# --------------------------------------------------------------------------------------


def test_a_run_carries_the_readings_on_either_side_of_it():
    values = np.array([0.31, 0.0, 0.0, 0.31, 1.00], dtype=np.float32)

    (run,) = sentinel_runs(values, _bounds(5))

    assert (run.start, run.length) == (1, 2)
    assert run.left == pytest.approx(0.31)
    assert run.right == pytest.approx(0.31)
    assert run.bracketed and run.brackets_agree and not run.leading


def test_a_run_that_opens_an_episode_has_nothing_behind_it():
    values = np.array([0.0, 0.0, 0.31], dtype=np.float32)

    (run,) = sentinel_runs(values, _bounds(3))

    assert run.leading and run.left is None
    assert run.right == pytest.approx(0.31)


def test_a_run_that_closes_an_episode_still_has_something_to_hold():
    values = np.array([0.31, 0.0, 0.0], dtype=np.float32)

    (run,) = sentinel_runs(values, _bounds(3))

    assert not run.leading and run.right is None


def test_the_last_reading_is_not_carried_across_an_episode_boundary():
    """Episode 1 starts fresh; holding episode 0's last width over it would invent a measurement."""
    values = np.array([0.31, 0.31, 0.0, 0.0, 0.90], dtype=np.float32)

    runs = sentinel_runs(values, _bounds(2, 3))

    assert [run.episode for run in runs] == [1]
    assert runs[0].leading


def test_episodes_that_are_interleaved_are_refused():
    with pytest.raises(RepairError, match="not contiguous"):
        episode_bounds(np.array([0, 1, 0]))


# --------------------------------------------------------------------------------------
# Repairing
# --------------------------------------------------------------------------------------


def test_the_repair_holds_the_last_reading_and_leaves_a_leading_run_alone():
    values = np.array([0.0, 0.31, 0.0, 0.0, 0.31], dtype=np.float32)
    runs = sentinel_runs(values, _bounds(5))

    repaired, filled, left = repair_values(values, runs, backfill_leading=False)

    assert repaired.tolist() == pytest.approx([0.0, 0.31, 0.31, 0.31, 0.31])
    assert (filled, left) == (2, 1)


def test_backfilling_a_leading_run_is_opt_in():
    values = np.array([0.0, 0.31, 0.0], dtype=np.float32)
    runs = sentinel_runs(values, _bounds(3))

    repaired, filled, left = repair_values(values, runs, backfill_leading=True)

    assert repaired.tolist() == pytest.approx([0.31, 0.31, 0.31])
    assert (filled, left) == (2, 0)


def test_the_repair_does_not_disturb_a_column_with_no_sentinels():
    values = np.array([0.31, 0.42, 0.99], dtype=np.float32)

    repaired, filled, left = repair_values(values, sentinel_runs(values, _bounds(3)), backfill_leading=False)

    assert np.array_equal(repaired, values)
    assert (filled, left) == (0, 0)


# --------------------------------------------------------------------------------------
# Deciding whether the repair applies at all
# --------------------------------------------------------------------------------------


def _audit_with(values: np.ndarray, bounds=None) -> Audit:
    from tools.fr3.fr3_repair_gripper_sentinel import column_stats

    bounds = bounds or _bounds(values.size)
    runs = sentinel_runs(values, bounds)
    repaired, filled, left = repair_values(values, runs, backfill_leading=False)
    return Audit(
        root=Path("ds"),
        frames=values.size,
        episodes=len(bounds),
        runs=runs,
        before=column_stats(values),
        after=column_stats(repaired),
        repaired_frames=filled,
        left_frames=left,
        orphan_episodes=[e for e, s, t in bounds if bool((values[s:t] == 0.0).all())],
    )


def _bracketed(level: float, other: float, runs: int) -> np.ndarray:
    """`runs` sentinel runs, each bracketed by `level` on the left and `other` on the right."""
    frames = [level]
    for _ in range(runs):
        frames += [0.0, 0.0, other, level]
    return np.array(frames, dtype=np.float32)


def test_a_recording_whose_readings_run_down_to_zero_is_refused():
    """A hand that really closes is recorded on its way down; a sentinel has nothing around it."""
    values = _bracketed(0.31, 0.31, MIN_BRACKETED_RUNS + 5)
    values[1] = SENTINEL_ISOLATION / 2

    (failure,) = premise_failures(_audit_with(values))

    assert "isolation band" in failure


def test_a_recording_where_the_hand_moved_through_the_gaps_is_refused():
    values = _bracketed(0.31, 0.95, MIN_BRACKETED_RUNS + 5)

    (failure,) = premise_failures(_audit_with(values))

    assert "come back to the level they left" in failure


def test_too_few_bracketed_runs_is_not_evidence():
    (failure,) = premise_failures(_audit_with(_bracketed(0.31, 0.31, 2)))

    assert f"need {MIN_BRACKETED_RUNS}" in failure


def test_a_recording_bracketed_by_the_same_level_throughout_is_ready():
    assert premise_failures(_audit_with(_bracketed(0.31, 0.31 + BRACKET_TOLERANCE / 2, 30))) == []


def test_a_healthy_recording_has_nothing_to_refuse():
    assert premise_failures(_audit_with(np.array([0.31, 0.42, 0.99], dtype=np.float32))) == []


def test_an_episode_that_is_sentinel_end_to_end_holds_nothing():
    values = np.concatenate([_bracketed(0.31, 0.31, 30), np.zeros(4, dtype=np.float32)])
    bounds = _bounds(values.size - 4, 4)

    failures = premise_failures(_audit_with(values, bounds))

    assert any("end to end" in failure for failure in failures)


# --------------------------------------------------------------------------------------
# Patching one dimension of a statistics block
# --------------------------------------------------------------------------------------


def _stats(vector: list[float], count: int = 4) -> dict[str, list]:
    return {key: ([count] if key == "count" else list(vector)) for key in STATS_KEYS}


def test_patching_writes_one_dimension_and_keeps_the_rest_bit_identical():
    stored = _stats([1.0, 0.0, 3.0])
    fresh = {key: np.array([1.0, 0.5, 3.0]) if key != "count" else np.array([4]) for key in STATS_KEYS}

    patched = patch_stats_vector(stored, fresh, 1, what="t")

    assert patched["mean"].tolist() == [1.0, 0.5, 3.0]
    assert patched["count"].tolist() == [4]


def test_patching_refuses_when_the_recomputation_moves_a_dimension_it_must_not():
    """A stats block half-written by two recipes is worse than a stale one: nothing can tell."""
    stored = _stats([1.0, 0.0, 3.0])
    fresh = {key: np.array([1.5, 0.5, 3.0]) if key != "count" else np.array([4]) for key in STATS_KEYS}

    with pytest.raises(RepairError, match="does not reproduce the stored value on dimension 0"):
        patch_stats_vector(stored, fresh, 1, what="t")


def test_patching_refuses_a_count_that_changed():
    stored = _stats([1.0, 0.0, 3.0], count=4)
    fresh = {key: np.array([1.0, 0.5, 3.0]) if key != "count" else np.array([5]) for key in STATS_KEYS}

    with pytest.raises(RepairError, match="count"):
        patch_stats_vector(stored, fresh, 1, what="t")


# --------------------------------------------------------------------------------------
# End to end, on a dataset shaped like the ones on the rig
# --------------------------------------------------------------------------------------


def _fake_stats(matrix: np.ndarray, axis=0, keepdims=False, quantile_list=None) -> dict[str, np.ndarray]:
    data = np.asarray(matrix, dtype=np.float64)
    return {
        "min": data.min(axis=0),
        "max": data.max(axis=0),
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
        "count": np.array([data.shape[0]]),
        **{key: np.percentile(data, int(key[1:]), axis=0) for key in ("q01", "q10", "q50", "q90", "q99")},
    }


def _fake_aggregate(stats_list):
    key = next(iter(stats_list[0]))
    counts = np.stack([s[key]["count"] for s in stats_list]).astype(np.float64)
    total = counts.sum(axis=0)
    out = {
        "min": np.min(np.stack([s[key]["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s[key]["max"] for s in stats_list]), axis=0),
        "count": total,
    }
    for stat in ("mean", "std", "q01", "q10", "q50", "q90", "q99"):
        values = np.stack([s[key][stat] for s in stats_list])
        out[stat] = (values * counts).sum(axis=0) / total
    return {key: out}


def _write_dataset(root: Path, gripper: np.ndarray, episode_lengths: list[int]) -> Path:
    """A LeRobot v3.0 recording, small but shaped exactly like the ones this repairs."""
    frames = int(sum(episode_lengths))
    assert gripper.size == frames
    episode_index = np.concatenate(
        [np.full(length, episode, dtype=np.int64) for episode, length in enumerate(episode_lengths)]
    )
    state = np.stack(
        [np.linspace(0.1, 0.2, frames), gripper, np.zeros(frames)], axis=1
    ).astype(np.float32)
    # The command channel is deliberately full of real zeros: 0.0 asks the hand to close.
    action = np.stack([np.linspace(0.0, 1.0, frames), np.zeros(frames)], axis=1).astype(np.float32)

    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    table = pa.table(
        {
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 2)),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 3)),
            "episode_index": pa.array(episode_index),
            "index": pa.array(np.arange(frames, dtype=np.int64)),
        }
    )
    # One row group per episode, the way the recorder writes them.
    with pq.ParquetWriter(root / "data" / "chunk-000" / "file-000.parquet", table.schema) as writer:
        start = 0
        for length in episode_lengths:
            writer.write_table(table.slice(start, length))
            start += length

    rows: dict[str, list] = {"episode_index": [], "dataset_from_index": [], "dataset_to_index": [], "length": []}
    rows.update({f"stats/observation.state/{key}": [] for key in STATS_KEYS})
    start = 0
    per_episode = []
    for episode, length in enumerate(episode_lengths):
        stats = _fake_stats(state[start : start + length])
        per_episode.append({"observation.state": stats})
        rows["episode_index"].append(episode)
        rows["dataset_from_index"].append(start)
        rows["dataset_to_index"].append(start + length)
        rows["length"].append(length)
        for key in STATS_KEYS:
            rows[f"stats/observation.state/{key}"].append(np.asarray(stats[key]).tolist())
        start += length
    pq.write_table(pa.table(rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    aggregated = _fake_aggregate(per_episode)["observation.state"]
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "action": {key: [0.0, 0.0] if key != "count" else [frames] for key in STATS_KEYS},
                "observation.state": {key: np.asarray(value).tolist() for key, value in aggregated.items()},
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": len(episode_lengths),
                "total_frames": frames,
                "features": {
                    "action": {"dtype": "float32", "shape": [2], "names": ["ee.x", "gripper.pos"]},
                    "observation.state": {"dtype": "float32", "shape": [3], "names": STATE_NAMES},
                },
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    return root


#: Two episodes shaped like the 2026-08-21 recordings: dropouts every few frames, each bracketed
#: by the level it left, and one run opening the first episode with nothing behind it. Long enough
#: that the bracket-agreement check has something to judge (MIN_BRACKETED_RUNS).
EPISODE_A = [0.0, 0.0] + [0.31, 0.0, 0.0] * 12 + [0.31]
EPISODE_B = [0.95] + [0.0, 0.0, 0.95] * 12
REPAIRED_A = [0.0, 0.0] + [0.31] * 37
REPAIRED_B = [0.95] * 37


@pytest.fixture
def dataset(tmp_path, monkeypatch) -> Path:
    import tools.fr3.fr3_repair_gripper_sentinel as tool

    monkeypatch.setattr(tool, "_lerobot_stats", lambda: (_fake_stats, _fake_aggregate))
    gripper = np.array(EPISODE_A + EPISODE_B, dtype=np.float32)
    return _write_dataset(tmp_path / "insert_20260821_000000", gripper, [len(EPISODE_A), len(EPISODE_B)])


def test_apply_repairs_the_gripper_and_leaves_every_other_number_alone(dataset):
    data_file = dataset / "data" / "chunk-000" / "file-000.parquet"
    before = pq.read_table(data_file)

    recording = read_recording(dataset)
    audit, repaired = audit_recording(recording, backfill_leading=False)
    record = apply_repair(recording, repaired, backfill_leading=False, state_key="observation.state")

    after = pq.read_table(data_file)
    assert after.schema == before.schema
    assert after.column("action").equals(before.column("action"))  # real zeros, never touched
    assert after.column("episode_index").equals(before.column("episode_index"))
    state = np.stack(after.column("observation.state").to_numpy(zero_copy_only=False))
    # Leading run left alone by default; everything after it holds the last reading.
    assert state[:, GRIPPER].tolist() == pytest.approx(REPAIRED_A + REPAIRED_B)
    unchanged = np.stack(before.column("observation.state").to_numpy(zero_copy_only=False))[:, 0]
    assert state[:, 0].tolist() == pytest.approx(unchanged.tolist())
    assert record["repaired_frames"] == 48  # 24 runs of two frames
    assert record["sentinel_frames_after"] == 2
    assert audit.left_frames == 2


def test_apply_keeps_one_row_group_per_episode(dataset):
    data_file = dataset / "data" / "chunk-000" / "file-000.parquet"
    groups = pq.ParquetFile(data_file).num_row_groups

    recording = read_recording(dataset)
    _, repaired = audit_recording(recording, backfill_leading=False)
    apply_repair(recording, repaired, backfill_leading=False, state_key="observation.state")

    assert pq.ParquetFile(data_file).num_row_groups == groups == 2


def test_apply_moves_the_statistics_only_on_the_gripper_dimension(dataset):
    stats_path = dataset / "meta" / "stats.json"
    before = json.loads(stats_path.read_text(encoding="utf-8"))
    episodes_before = pq.read_table(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    recording = read_recording(dataset)
    _, repaired = audit_recording(recording, backfill_leading=False)
    apply_repair(recording, repaired, backfill_leading=False, state_key="observation.state")

    after = json.loads(stats_path.read_text(encoding="utf-8"))
    assert after["action"] == before["action"]
    for key in STATS_KEYS:
        old = np.asarray(before["observation.state"][key], dtype=np.float64)
        new = np.asarray(after["observation.state"][key], dtype=np.float64)
        untouched = [i for i in range(old.size) if i != GRIPPER or key == "count"]
        assert new[untouched].tolist() == pytest.approx(old[untouched].tolist())
    # The mean can only rise: sentinels were the smallest value the column held.
    assert after["observation.state"]["mean"][GRIPPER] > before["observation.state"]["mean"][GRIPPER]
    assert after["observation.state"]["min"][GRIPPER] == pytest.approx(0.0)  # the leading run survives

    episodes_after = pq.read_table(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert episodes_after.schema == episodes_before.schema
    assert episodes_after.column("length").equals(episodes_before.column("length"))
    # Episode 1 had no leading run, so nothing of its gripper column is still a sentinel.
    assert episodes_after.column("stats/observation.state/min").to_pylist()[1][GRIPPER] == pytest.approx(0.95)


def test_apply_keeps_the_original_recoverable_and_never_overwrites_it(dataset):
    data_file = dataset / "data" / "chunk-000" / "file-000.parquet"
    original = pq.read_table(data_file)

    recording = read_recording(dataset)
    _, repaired = audit_recording(recording, backfill_leading=False)
    apply_repair(recording, repaired, backfill_leading=False, state_key="observation.state")
    backup = dataset / "meta" / REPAIR_DIRNAME / "original" / "data" / "chunk-000" / "file-000.parquet"
    assert pq.read_table(backup).equals(original)

    # A second pass, this time filling the leading run, must not overwrite the first backup.
    recording = read_recording(dataset)
    _, repaired = audit_recording(recording, backfill_leading=True)
    apply_repair(recording, repaired, backfill_leading=True, state_key="observation.state")

    assert pq.read_table(backup).equals(original)
    assert [r["backfill_leading"] for r in read_provenance(dataset)["repairs"]] == [False, True]
    state = np.stack(pq.read_table(data_file).column("observation.state").to_numpy(zero_copy_only=False))
    assert 0.0 not in state[:, GRIPPER].tolist()


def test_the_backup_is_not_where_lerobot_looks_for_data(dataset):
    """LeRobot loads `data/*/*.parquet` and `meta/episodes/*/*.parquet`; a backup in either is data."""
    recording = read_recording(dataset)
    _, repaired = audit_recording(recording, backfill_leading=False)
    apply_repair(recording, repaired, backfill_leading=False, state_key="observation.state")

    assert len(sorted((dataset / "data").glob("*/*.parquet"))) == 1
    assert len(sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))) == 1


def test_verification_catches_a_gripper_column_that_is_not_the_repaired_one(dataset):
    recording = read_recording(dataset)
    _, repaired = audit_recording(recording, backfill_leading=False)
    apply_repair(recording, repaired, backfill_leading=False, state_key="observation.state")

    with pytest.raises(RepairError, match="not the repaired one"):
        verify_written(read_recording(dataset), repaired + 0.5, state_key="observation.state")


def test_a_report_run_writes_nothing(dataset, capsys):
    data_file = dataset / "data" / "chunk-000" / "file-000.parquet"
    before = data_file.read_bytes()

    assert main([str(dataset)]) == 0

    out = capsys.readouterr().out
    assert data_file.read_bytes() == before
    assert "nothing written" in out
    assert "would repair" in out
    assert not (dataset / "meta" / REPAIR_DIRNAME).exists()


def test_the_report_refuses_out_loud_when_a_zero_could_be_a_real_reading(tmp_path, capsys):
    gripper = np.array([0.31, 0.0, 0.0, 0.31, SENTINEL_ISOLATION / 2, 0.31], dtype=np.float32)
    root = _write_dataset(tmp_path / "ambiguous", gripper, [gripper.size])

    assert main([str(root)]) == 0
    assert "REFUSED" in capsys.readouterr().out

    with pytest.raises(RepairError, match="refusing to write"):
        main([str(root), "--apply"])


def test_views_built_on_a_repaired_recording_are_named_as_stale(tmp_path):
    """`source_digest` hashes paths and settings, not bytes, so nothing else will notice."""
    recording = tmp_path / "datasets" / "insert_20260821_000000"
    views = tmp_path / "training_views"
    for name, sources in (("uses_it", [str(recording)]), ("does_not", ["/elsewhere/other"])):
        manifest = views / name / "meta" / "il_view_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"source_dataset_roots": sources}), encoding="utf-8")

    assert [path.name for path in stale_views([recording], views)] == ["uses_it"]
    assert stale_views([recording], tmp_path / "absent") == []
