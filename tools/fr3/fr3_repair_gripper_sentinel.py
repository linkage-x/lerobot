#!/usr/bin/env python3
"""Replace the Pika SDK's sentinel 0.0 in a recording's gripper observation with the last real reading.

``PikaGripperHardwareDriver.get_position()`` is ``clip(get_gripper_distance() / max_width, 0, 1)``
with no freshness check, and the SDK keeps returning its initial ``0.0`` from
``get_gripper_distance()`` until it parses a frame carrying ``motor``. A recording made while that
link is unhealthy therefore stores the sentinel as if it were a measured width. On 2026-08-21 that
happened on ~47% of frames, and the whole ``insert`` demonstration corpus was recorded that day --
so every model trained since has seen a policy *input* that the live rig never produces.

The repair is "hold the last valid reading", which is what the driver should have done. Run without
``--apply`` it changes nothing and prints the evidence for and against repairing each dataset; the
same report on a healthy recording is the comparison row, because a healthy recording has no
sentinels to find.

Four things this refuses to do, because each one would turn a repair into a corruption:

* **Touch a command.** ``action:gripper.pos`` and ``observation.state:prev_cmd.gripper.pos`` carry
  ~43% zeros on the same recordings and every one of them is real: 0.0 asks the hand to close. Only
  the measured column is rewritten, and the write is verified afterwards against the original.

* **Repair a dataset where an exact 0 could be a real reading.** A hand that really closes passes
  through the intermediate widths on its way, so a sentinel is recognisable by being *isolated* --
  nothing between it and the smallest genuine reading. Where the readings run down to 0 in a
  continuum instead, this refuses rather than guessing (see ``SENTINEL_ISOLATION``).

* **Guess across an episode boundary.** "The last valid reading" means the last one in the same
  episode. A run of sentinels that opens an episode has nothing behind it, so by default it is
  left alone and counted -- ``--backfill-leading`` fills it from the first valid reading ahead,
  which is a different assumption and is therefore opt-in.

* **Leave the statistics describing data that is no longer there.** ``meta/stats.json`` and the
  per-episode ``stats/observation.state/*`` columns feed normalization. Both are recomputed with
  LeRobot's own functions, and only the gripper dimension is written back -- every other dimension
  has to still agree with what is on disk, or the run is refused, because a disagreement would mean
  this tool no longer understands how those numbers were produced.

Rebuilding the training views afterwards is a separate, explicit step: ``source_digest`` in
``meta/il_view_manifest.json`` hashes the source *paths and settings*, not their contents, so a
view built from an unrepaired recording will not notice that its inputs changed. The report lists
the views that go stale.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIEWS_ROOT = ROOT / "outputs" / "exports" / "training_views"

#: The feature holding the measured gripper width, and its entry in that feature's `names`.
STATE_KEY = "observation.state"
GRIPPER_NAME = "gripper.pos"

#: The value the Pika SDK hands out before it has parsed anything. Exact, not approximate: it is a
#: fresh float, never a scaled measurement, so it arrives bit-identical every time.
SENTINEL = 0.0

#: How far the smallest genuine reading has to sit above the sentinel before an exact 0 counts as
#: distinguishable from a closed hand. The hand cannot teleport shut: a real close is recorded as a
#: descent through the intermediate widths, so a gap this wide with nothing in it is the signature
#: of a value that was never measured. 0.05 of full travel is ~4 mm on this hand -- several frames'
#: worth of motion at any speed the arm actually closes at.
SENTINEL_ISOLATION = 0.05

#: Two readings this close are the same physical width. Used only to ask whether a run of sentinels
#: is bracketed by the same level on both sides, which is the evidence that nothing moved while the
#: link was down.
BRACKET_TOLERANCE = 0.02

#: How much of that bracketing evidence has to be there before "hold the last reading" is the right
#: repair. Runs that span a genuine open or close will disagree, so this is not expected to reach
#: 1.0 -- on the 2026-08-21 recordings it sits at 92%.
MIN_BRACKET_AGREEMENT = 0.80

#: Below this many bracketed runs the agreement rate is not evidence of anything, and the check is
#: reported as inconclusive rather than passed.
MIN_BRACKETED_RUNS = 20

#: Statistics LeRobot keeps per feature, in the order it writes them.
STATS_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")

#: Recomputed stats have to reproduce the stored ones on every dimension the repair did not touch.
#: Loose enough for float64 accumulation order, far tighter than any real disagreement.
STATS_TOLERANCE = 1e-6

REPAIR_DIRNAME = "gripper_sentinel_repair"
PROVENANCE_NAME = "repair.json"


class RepairError(RuntimeError):
    """A refusal the operator can fix."""


@dataclass(frozen=True)
class Run:
    """A maximal block of sentinel readings inside one episode."""

    episode: int
    start: int
    length: int
    #: The last genuine reading before the run, and the first one after it, within this episode.
    #: `None` means the run touches that end of the episode and has nothing to hold.
    left: float | None
    right: float | None

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def leading(self) -> bool:
        return self.left is None

    @property
    def bracketed(self) -> bool:
        return self.left is not None and self.right is not None

    @property
    def brackets_agree(self) -> bool:
        return self.bracketed and abs(self.left - self.right) <= BRACKET_TOLERANCE


@dataclass
class Audit:
    """What one dataset's gripper column looks like, and whether the repair applies to it."""

    root: Path
    frames: int
    episodes: int
    runs: list[Run] = field(default_factory=list)
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    repaired_frames: int = 0
    left_frames: int = 0
    orphan_episodes: list[int] = field(default_factory=list)
    already_repaired: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sentinel_frames(self) -> int:
        return sum(run.length for run in self.runs)

    @property
    def bracketed(self) -> list[Run]:
        return [run for run in self.runs if run.bracketed]

    @property
    def bracket_agreement(self) -> float:
        bracketed = self.bracketed
        if not bracketed:
            return float("nan")
        return sum(1 for run in bracketed if run.brackets_agree) / len(bracketed)


def episode_bounds(episode_index: np.ndarray) -> list[tuple[int, int, int]]:
    """`(episode, start, stop)` for each episode, refusing a file whose episodes are interleaved.

    "The last valid reading" only means anything inside one episode, so an episode that appears in
    two separate blocks would let a value be held across frames that never followed each other.
    """
    if episode_index.size == 0:
        return []
    edges = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [episode_index.size]))
    blocks = [(int(episode_index[start]), int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]
    seen = [episode for episode, _, _ in blocks]
    if len(set(seen)) != len(seen):
        raise RepairError("episode_index is not contiguous; the rows are out of order or interleaved")
    return blocks


def sentinel_runs(values: np.ndarray, bounds: Sequence[tuple[int, int, int]]) -> list[Run]:
    """Every maximal block of exact sentinels, tagged with the genuine readings around it."""
    runs: list[Run] = []
    for episode, start, stop in bounds:
        block = values[start:stop]
        is_sentinel = (block == SENTINEL).astype(np.int8)
        if not is_sentinel.any():
            continue
        # Pad with a non-sentinel on both sides so a run touching either end still has an edge.
        edges = np.diff(np.concatenate(([0], is_sentinel, [0])))
        for run_start, run_stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            run_start, run_stop = int(run_start), int(run_stop)
            runs.append(
                Run(
                    episode=episode,
                    start=start + run_start,
                    length=run_stop - run_start,
                    left=None if run_start == 0 else float(block[run_start - 1]),
                    right=None if run_stop == block.size else float(block[run_stop]),
                )
            )
    return runs


def repair_values(
    values: np.ndarray, runs: Sequence[Run], *, backfill_leading: bool
) -> tuple[np.ndarray, int, int]:
    """Hold the last genuine reading over each run. Returns the new column and (repaired, left).

    A leading run has nothing behind it. Filling it from the reading *ahead* is a second assumption
    -- that the hand had not moved yet when recording started -- so it happens only when asked for.
    """
    repaired = values.copy()
    filled = left = 0
    for run in runs:
        source = run.left if run.left is not None else (run.right if backfill_leading else None)
        if source is None:
            left += run.length
            continue
        repaired[run.start : run.stop] = np.asarray(source, dtype=values.dtype)
        filled += run.length
    return repaired, filled, left


def column_stats(values: np.ndarray) -> dict[str, float]:
    """A description of the gripper column terse enough to put beside another dataset's."""
    finite = np.asarray(values, dtype=np.float64)
    genuine = finite[finite != SENTINEL]
    return {
        "frames": int(finite.size),
        "sentinel_frac": float((finite == SENTINEL).mean()) if finite.size else float("nan"),
        "genuine_min": float(genuine.min()) if genuine.size else float("nan"),
        "q01": float(np.percentile(finite, 1)) if finite.size else float("nan"),
        "q50": float(np.percentile(finite, 50)) if finite.size else float("nan"),
        "q99": float(np.percentile(finite, 99)) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std()) if finite.size else float("nan"),
    }


def premise_failures(audit: Audit) -> list[str]:
    """Why this dataset must not be repaired, empty when it may be. Each entry names its evidence."""
    failures: list[str] = []
    if not audit.runs:
        return failures

    genuine_min = audit.before.get("genuine_min", float("nan"))
    if np.isnan(genuine_min):
        failures.append("the gripper column has no genuine reading at all")
    elif genuine_min < SENTINEL_ISOLATION:
        failures.append(
            f"readings run down to {genuine_min:.4f}, inside the {SENTINEL_ISOLATION} isolation "
            "band, so an exact 0 here is not distinguishable from a closed hand"
        )

    bracketed = audit.bracketed
    if len(bracketed) < MIN_BRACKETED_RUNS:
        failures.append(
            f"only {len(bracketed)} sentinel runs have a reading on both sides (need "
            f"{MIN_BRACKETED_RUNS}); there is not enough evidence that nothing moved during them"
        )
    elif audit.bracket_agreement < MIN_BRACKET_AGREEMENT:
        failures.append(
            f"only {audit.bracket_agreement:.1%} of bracketed runs come back to the level they "
            f"left (need {MIN_BRACKET_AGREEMENT:.0%}); the hand was moving through these, and "
            "holding the last reading would invent a hold that did not happen"
        )

    if audit.orphan_episodes:
        failures.append(
            f"episodes {audit.orphan_episodes[:5]} are sentinel from end to end and hold nothing"
        )
    return failures


def as_matrix(column: Any) -> np.ndarray:
    """A `fixed_size_list` column as an (n, dim) float32 array."""
    values = column.to_numpy(zero_copy_only=False)
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(values).astype(np.float32, copy=False)


def _lerobot_stats():
    """LeRobot's own statistics functions, imported from *this* checkout.

    The numbers on disk were produced by these functions -- histogram quantiles, count-weighted
    aggregation -- and the repair overwrites one dimension of them. Recomputing that dimension some
    other way would leave a stats block whose entries disagree about what they mean. A sibling
    clone earlier on `sys.path` would satisfy the import and answer for a different version, so
    where the module actually came from is checked rather than assumed.
    """
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from lerobot.datasets import compute_stats
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RepairError(f"cannot import LeRobot from {source}: {exc}") from exc
    resolved = Path(compute_stats.__file__).resolve()
    if ROOT not in resolved.parents:
        raise RepairError(
            f"lerobot resolved to {resolved}, outside this checkout ({ROOT}). Re-run with "
            f"PYTHONPATH={ROOT / 'src'} so the stats are recomputed by the code that wrote them."
        )
    return compute_stats.get_feature_stats, compute_stats.aggregate_stats


@dataclass
class Recording:
    """One LeRobot v3.0 recording, opened far enough to audit and repair its gripper column."""

    root: Path
    gripper_index: int
    data_files: list[Path]
    rows_per_file: list[int]
    state: np.ndarray
    episode_index: np.ndarray
    episode_files: list[Path]

    @property
    def gripper(self) -> np.ndarray:
        return self.state[:, self.gripper_index]


def read_recording(root: Path, *, state_key: str = STATE_KEY, gripper_name: str = GRIPPER_NAME) -> Recording:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise RepairError(f"{root} is not a LeRobot dataset: no meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = (info.get("features") or {}).get(state_key)
    if not feature:
        raise RepairError(f"{root} has no '{state_key}' feature")
    names = list(feature.get("names") or [])
    if gripper_name not in names:
        raise RepairError(f"{root}: '{state_key}' has no '{gripper_name}' column (names: {names})")

    # The same glob, in the same order, that LeRobot loads the rows in -- so the row offsets here
    # are the `dataset_from_index` offsets the episode metadata is written against.
    data_files = sorted((root / "data").glob("*/*.parquet"))
    if not data_files:
        raise RepairError(f"{root} has no data parquet under data/*/*.parquet")
    states: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    rows_per_file: list[int] = []
    for path in data_files:
        table = pq.read_table(path, columns=[state_key, "episode_index"])
        if table.column(state_key).null_count:
            raise RepairError(f"{path}: '{state_key}' contains nulls, which this tool cannot preserve")
        states.append(as_matrix(table.column(state_key)))
        episodes.append(table.column("episode_index").to_numpy())
        rows_per_file.append(table.num_rows)

    return Recording(
        root=root,
        gripper_index=names.index(gripper_name),
        data_files=data_files,
        rows_per_file=rows_per_file,
        state=np.concatenate(states) if states else np.empty((0, len(names)), dtype=np.float32),
        episode_index=np.concatenate(episodes) if episodes else np.empty(0, dtype=np.int64),
        episode_files=sorted((root / "meta" / "episodes").glob("*/*.parquet")),
    )


def audit_recording(recording: Recording, *, backfill_leading: bool) -> tuple[Audit, np.ndarray]:
    """Describe the gripper column and what the repair would do to it, without writing anything."""
    bounds = episode_bounds(recording.episode_index)
    gripper = recording.gripper
    runs = sentinel_runs(gripper, bounds)
    repaired, filled, left = repair_values(gripper, runs, backfill_leading=backfill_leading)
    orphans = [
        episode
        for episode, start, stop in bounds
        if stop > start and bool((gripper[start:stop] == SENTINEL).all())
    ]
    audit = Audit(
        root=recording.root,
        frames=int(gripper.size),
        episodes=len(bounds),
        runs=runs,
        before=column_stats(gripper),
        after=column_stats(repaired),
        repaired_frames=filled,
        left_frames=left,
        orphan_episodes=orphans,
        already_repaired=read_provenance(recording.root).get("repairs", []),
    )
    return audit, repaired


def patch_stats_vector(
    stored: dict[str, Any], fresh: dict[str, Any], index: int, *, what: str
) -> dict[str, np.ndarray]:
    """Overwrite one dimension of a stored stats block with a freshly computed one.

    Every other dimension has to already agree. The stored numbers came from the same functions
    over the same rows, so a disagreement anywhere else means this tool is no longer reproducing
    how they were made -- and a stats block half-computed by two different recipes is worse than
    one that is merely out of date, because nothing downstream can tell.
    """
    patched: dict[str, np.ndarray] = {}
    for key in STATS_KEYS:
        if key not in stored or key not in fresh:
            raise RepairError(f"{what}: stats are missing '{key}'")
        old_raw = np.asarray(stored[key])
        new = np.asarray(fresh[key], dtype=np.float64)
        if old_raw.shape != new.shape:
            raise RepairError(f"{what}: stored '{key}' has shape {old_raw.shape}, recomputed {new.shape}")
        if key == "count":
            if not np.array_equal(old_raw.astype(np.int64), new.astype(np.int64)):
                raise RepairError(f"{what}: stored count {old_raw.tolist()} != recomputed {new.tolist()}")
            patched[key] = old_raw
            continue
        old = old_raw.astype(np.float64)
        untouched = np.ones(old.shape, dtype=bool)
        untouched[index] = False
        drift = np.abs(old - new)[untouched]
        if drift.size and float(drift.max()) > STATS_TOLERANCE:
            worst = int(np.argmax(np.where(untouched, np.abs(old - new), -np.inf)))
            raise RepairError(
                f"{what}: recomputing '{key}' does not reproduce the stored value on dimension "
                f"{worst}, which this repair does not touch (stored {old[worst]!r}, recomputed "
                f"{new[worst]!r}). Refusing to write a stats block built from two recipes."
            )
        merged = old.copy()
        merged[index] = new[index]
        patched[key] = merged
    return patched


def _json_indent(text: str) -> int:
    """The indent the file already uses, so a one-dimension repair stays a one-dimension diff."""
    for line in text.splitlines()[1:]:
        stripped = line.lstrip(" ")
        if stripped and stripped != line:
            return len(line) - len(stripped)
    return 2


def read_provenance(root: Path) -> dict[str, Any]:
    path = root / "meta" / REPAIR_DIRNAME / PROVENANCE_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def back_up(root: Path, path: Path) -> Path:
    """Copy a file into the repair directory once, and never over an existing copy.

    A second repair pass -- filling the leading runs after a first pass held the rest -- must still
    leave the *original* recoverable, so the first backup is the one that is kept.
    """
    target = root / "meta" / REPAIR_DIRNAME / "original" / path.relative_to(root)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return target


def rewrite_data_file(path: Path, new_state: np.ndarray, *, state_key: str = STATE_KEY) -> None:
    """Rewrite one data parquet with a new state column, preserving everything else about it.

    Row groups are written back one for one. LeRobot selects episodes with a pyarrow predicate,
    which reads at row-group granularity, and the recorder writes one row group per episode; a
    rewrite that coalesced them would make every episode read cost the whole file.
    """
    source = pq.ParquetFile(path)
    schema = source.schema_arrow
    index = schema.get_field_index(state_key)
    if index < 0:
        raise RepairError(f"{path} has no '{state_key}' column")
    field_ = schema.field(index)
    compression = source.metadata.row_group(0).column(0).compression.lower()
    tmp = path.with_name(path.name + ".repair-tmp")
    offset = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression=compression) as writer:
            for group in range(source.num_row_groups):
                table = source.read_row_group(group)
                chunk = new_state[offset : offset + table.num_rows]
                offset += table.num_rows
                column = pa.array([row.tolist() for row in chunk], type=field_.type)
                writer.write_table(table.set_column(index, field_, column))
        if offset != new_state.shape[0]:
            raise RepairError(f"{path}: wrote {offset} rows but was given {new_state.shape[0]}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def rewrite_episode_stats(path: Path, patched: dict[int, dict[str, np.ndarray]], *, state_key: str) -> None:
    source = pq.ParquetFile(path)
    schema = source.schema_arrow
    table = source.read()
    compression = source.metadata.row_group(0).column(0).compression.lower()
    for key in STATS_KEYS:
        name = f"stats/{state_key}/{key}"
        index = schema.get_field_index(name)
        if index < 0:
            raise RepairError(f"{path} has no '{name}' column")
        field_ = schema.field(index)
        rows = [np.asarray(patched[int(episode)][key]).tolist() for episode in table.column("episode_index").to_pylist()]
        table = table.set_column(index, field_, pa.array(rows, type=field_.type))
    tmp = path.with_name(path.name + ".repair-tmp")
    try:
        pq.write_table(table, tmp, compression=compression)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def episode_ranges(recording: Recording) -> list[tuple[int, int, int, Path]]:
    """`(episode, from_index, to_index, meta_file)` as the episode metadata declares them.

    Checked against the rows rather than trusted: the stats about to be rewritten are attached to
    these ranges, and a range that does not hold the episode it claims would move one episode's
    numbers onto another's.
    """
    if not recording.episode_files:
        raise RepairError(f"{recording.root} has no meta/episodes/*/*.parquet")
    ranges: list[tuple[int, int, int, Path]] = []
    for path in recording.episode_files:
        table = pq.read_table(path, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
        for episode, start, stop in zip(
            table.column("episode_index").to_pylist(),
            table.column("dataset_from_index").to_pylist(),
            table.column("dataset_to_index").to_pylist(),
            strict=True,
        ):
            rows = recording.episode_index[int(start) : int(stop)]
            if rows.size == 0 or not bool((rows == episode).all()):
                raise RepairError(
                    f"{recording.root}: episode {episode} claims rows [{start}, {stop}) but those "
                    "rows do not all belong to it"
                )
            ranges.append((int(episode), int(start), int(stop), path))
    covered = sum(stop - start for _, start, stop, _ in ranges)
    if covered != recording.state.shape[0]:
        raise RepairError(
            f"{recording.root}: episode ranges cover {covered} rows, the data files hold "
            f"{recording.state.shape[0]}"
        )
    return ranges


def verify_written(recording: Recording, repaired: np.ndarray, *, state_key: str) -> None:
    """Read the files back and prove only the gripper dimension moved, and only where it had to.

    This is the check that the commands were left alone. `action:gripper.pos` and
    `observation.state:prev_cmd.gripper.pos` are full of real zeros -- 0.0 asks the hand to close --
    and repairing one of them would rewrite what the operator did.
    """
    offset = 0
    for path in recording.data_files:
        backup = recording.root / "meta" / REPAIR_DIRNAME / "original" / path.relative_to(recording.root)
        before, after = pq.read_table(backup), pq.read_table(path)
        if before.schema != after.schema:
            raise RepairError(f"{path}: the rewrite changed the schema")
        for name in before.column_names:
            if name != state_key and not before.column(name).equals(after.column(name)):
                raise RepairError(f"{path}: the rewrite changed column '{name}', which it must not touch")
        rows = after.num_rows
        old, new = as_matrix(before.column(state_key)), as_matrix(after.column(state_key))
        moved = np.argwhere(old != new)
        if moved.size and set(moved[:, 1].tolist()) != {recording.gripper_index}:
            other = sorted(set(moved[:, 1].tolist()) - {recording.gripper_index})
            raise RepairError(f"{path}: the rewrite changed state dimensions {other}, not just the gripper")
        expected = repaired[offset : offset + rows]
        if not np.array_equal(new[:, recording.gripper_index], expected):
            raise RepairError(f"{path}: the gripper column on disk is not the repaired one")
        offset += rows


def stale_views(roots: Sequence[Path], views_root: Path) -> list[Path]:
    """Training views whose manifest names one of these recordings.

    `source_digest` hashes the source paths and the merge settings, not the bytes underneath them,
    so a view built before the repair will keep looking current. Nothing downstream will notice;
    the rebuild has to be asked for.
    """
    if not views_root.is_dir():
        return []
    wanted = {str(root.resolve()) for root in roots}
    stale: list[Path] = []
    for manifest in sorted(views_root.glob("*/meta/il_view_manifest.json")):
        text = manifest.read_text(encoding="utf-8")
        if any(root in text for root in wanted):
            stale.append(manifest.parent.parent)
    return stale


def apply_repair(
    recording: Recording, repaired: np.ndarray, *, backfill_leading: bool, state_key: str
) -> dict[str, Any]:
    """Write the repaired column and the statistics that describe it. Backups first, verify last."""
    get_feature_stats, aggregate = _lerobot_stats()
    ranges = episode_ranges(recording)

    for path in [*recording.data_files, *recording.episode_files, recording.root / "meta" / "stats.json"]:
        if path.exists():
            back_up(recording.root, path)

    new_state = recording.state.copy()
    new_state[:, recording.gripper_index] = repaired

    # Per-episode stats first: the global block is their aggregate, so a failure here stops the run
    # before anything on disk has been touched.
    stored_episode_stats: dict[int, dict[str, Any]] = {}
    for path in recording.episode_files:
        table = pq.read_table(path)
        for row in table.to_pylist():
            stored_episode_stats[int(row["episode_index"])] = {
                key: row[f"stats/{state_key}/{key}"] for key in STATS_KEYS
            }
    patched_episode_stats: dict[int, dict[str, np.ndarray]] = {}
    for episode, start, stop, _ in ranges:
        fresh = get_feature_stats(new_state[start:stop], axis=0, keepdims=False)
        patched_episode_stats[episode] = patch_stats_vector(
            stored_episode_stats[episode],
            fresh,
            recording.gripper_index,
            what=f"{recording.root.name} episode {episode}",
        )

    stats_path = recording.root / "meta" / "stats.json"
    stats_text = stats_path.read_text(encoding="utf-8")
    stats_all = json.loads(stats_text)
    aggregated = aggregate(
        [{state_key: patched_episode_stats[episode]} for episode, _, _, _ in ranges]
    )[state_key]
    patched_global = patch_stats_vector(
        stats_all[state_key], aggregated, recording.gripper_index, what=f"{recording.root.name} meta/stats.json"
    )

    offset = 0
    for path, rows in zip(recording.data_files, recording.rows_per_file, strict=True):
        rewrite_data_file(path, new_state[offset : offset + rows], state_key=state_key)
        offset += rows
    for path in recording.episode_files:
        rewrite_episode_stats(path, patched_episode_stats, state_key=state_key)
    stats_all[state_key] = {key: np.asarray(value).tolist() for key, value in patched_global.items()}
    stats_path.write_text(json.dumps(stats_all, indent=_json_indent(stats_text)) + "\n", encoding="utf-8")

    verify_written(recording, repaired, state_key=state_key)

    record = {
        "applied_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool": Path(__file__).name,
        "rule": "hold the last valid reading within the episode",
        "backfill_leading": bool(backfill_leading),
        "state_key": state_key,
        "gripper_index": recording.gripper_index,
        "frames": int(repaired.size),
        "repaired_frames": int(np.count_nonzero(recording.gripper != repaired)),
        "sentinel_frames_before": int(np.count_nonzero(recording.gripper == SENTINEL)),
        "sentinel_frames_after": int(np.count_nonzero(repaired == SENTINEL)),
        "original_backup": str((recording.root / "meta" / REPAIR_DIRNAME / "original").relative_to(recording.root)),
    }
    provenance = read_provenance(recording.root)
    provenance.setdefault("repairs", []).append(record)
    provenance_path = recording.root / "meta" / REPAIR_DIRNAME / PROVENANCE_NAME
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return record


def summarise(audit: Audit) -> dict[str, Any]:
    lengths = [run.length for run in audit.runs]
    leading = [run for run in audit.runs if run.leading]
    trailing = [run for run in audit.runs if run.right is None]
    return {
        "dataset": audit.root.name,
        "root": str(audit.root),
        "frames": audit.frames,
        "episodes": audit.episodes,
        "sentinel_frames": audit.sentinel_frames,
        "sentinel_frac": audit.before["sentinel_frac"],
        "runs": len(audit.runs),
        "run_len_median": float(np.median(lengths)) if lengths else float("nan"),
        "run_len_p90": float(np.percentile(lengths, 90)) if lengths else float("nan"),
        "run_len_max": max(lengths) if lengths else 0,
        "leading_runs": len(leading),
        "leading_frames": sum(run.length for run in leading),
        "trailing_runs": len(trailing),
        "bracketed_runs": len(audit.bracketed),
        "bracket_agreement": audit.bracket_agreement,
        "genuine_min": audit.before["genuine_min"],
        "before": audit.before,
        "after": audit.after,
        "repaired_frames": audit.repaired_frames,
        "left_frames": audit.left_frames,
        "orphan_episodes": audit.orphan_episodes,
        "already_repaired": audit.already_repaired,
        "premise_failures": premise_failures(audit),
    }


def format_report(summaries: Sequence[dict[str, Any]]) -> str:
    rows: list[tuple[str, ...]] = [("", *[s["dataset"] for s in summaries])]

    def add(name: str, render) -> None:
        rows.append((name, *[render(s) for s in summaries]))

    add("frames", lambda s: f"{s['frames']}")
    add("episodes", lambda s: f"{s['episodes']}")
    add("sentinel frames", lambda s: f"{s['sentinel_frames']} ({s['sentinel_frac']:.2%})")
    add("sentinel runs", lambda s: f"{s['runs']}")
    add(
        "run length (med/p90/max)",
        lambda s: "-" if not s["runs"] else f"{s['run_len_median']:.0f}/{s['run_len_p90']:.0f}/{s['run_len_max']}",
    )
    add("runs opening an episode", lambda s: f"{s['leading_runs']} ({s['leading_frames']} frames)")
    add("runs closing an episode", lambda s: f"{s['trailing_runs']}")
    add(
        f"brackets agree (+-{BRACKET_TOLERANCE})",
        lambda s: "-" if not s["bracketed_runs"] else f"{s['bracket_agreement']:.1%} of {s['bracketed_runs']}",
    )
    add("smallest genuine reading", lambda s: f"{s['genuine_min']:.4f}")
    add("gripper q01/q50/q99 now", lambda s: _quantiles(s["before"]))
    add("gripper q01/q50/q99 after", lambda s: _quantiles(s["after"]))
    add("would repair", lambda s: f"{s['repaired_frames']} frames")
    add("would leave", lambda s: f"{s['left_frames']} frames" if s["left_frames"] else "-")
    add(
        "verdict",
        lambda s: "nothing to repair" if not s["runs"] else ("REFUSED" if s["premise_failures"] else "ready"),
    )

    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))

    for summary in summaries:
        for failure in summary["premise_failures"]:
            lines.append(f"\nREFUSED {summary['dataset']}: {failure}")
        for record in summary["already_repaired"]:
            lines.append(
                f"\nnote    {summary['dataset']}: already repaired at {record.get('applied_at')} "
                f"({record.get('repaired_frames')} frames); the original is under "
                f"meta/{REPAIR_DIRNAME}/original/"
            )
    return "\n".join(lines)


def _quantiles(stats: dict[str, float]) -> str:
    return f"{stats['q01']:.3f}/{stats['q50']:.3f}/{stats['q99']:.3f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", type=Path, help="LeRobot v3.0 recording roots.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the repair. Without it nothing is modified and the report is the whole output.",
    )
    parser.add_argument(
        "--backfill-leading",
        action="store_true",
        help="Also fill runs that open an episode, from the first genuine reading ahead of them. "
        "Off by default: holding the last reading is the rule the repair was validated as, and "
        "reaching backwards is a second assumption about a hand nobody had commanded yet.",
    )
    parser.add_argument("--state-key", default=STATE_KEY)
    parser.add_argument("--gripper-name", default=GRIPPER_NAME)
    parser.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply despite a failed premise check or an earlier repair. The checks are the "
        "argument that these zeros were never measured; say why in the commit message.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the summaries as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audits: list[tuple[Recording, Audit, np.ndarray]] = []
    for root in args.datasets:
        recording = read_recording(root, state_key=args.state_key, gripper_name=args.gripper_name)
        audit, repaired = audit_recording(recording, backfill_leading=args.backfill_leading)
        audits.append((recording, audit, repaired))

    summaries = [summarise(audit) for _, audit, _ in audits]
    if args.json:
        print(json.dumps({"datasets": summaries}, indent=2, default=str))
    else:
        print(format_report(summaries))

    if not args.apply:
        actionable = [s for s in summaries if s["runs"] and not s["premise_failures"]]
        if actionable and not args.json:
            print(f"\nnothing written. Re-run with --apply to repair {len(actionable)} dataset(s).")
        return 0

    blocked = [s for s in summaries if s["premise_failures"] or s["already_repaired"]]
    if blocked and not args.force:
        raise RepairError(
            "refusing to write: "
            + "; ".join(
                f"{s['dataset']} ({'; '.join(s['premise_failures']) or 'already repaired'})" for s in blocked
            )
        )

    written: list[Path] = []
    for recording, audit, repaired in audits:
        if not audit.runs:
            continue
        record = apply_repair(
            recording,
            repaired,
            backfill_leading=args.backfill_leading,
            state_key=args.state_key,
        )
        written.append(recording.root)
        print(
            f"repaired {recording.root.name}: {record['repaired_frames']} frames, "
            f"{record['sentinel_frames_after']} sentinel frames left"
        )

    for view in stale_views(written, args.views_root):
        print(f"stale   {view} -- built from a repaired recording; rebuild it before training on it")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RepairError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
