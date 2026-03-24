#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

from tools.fr3 import fr3_compare_live_capture_to_dataset_runtime as compare_runtime


class _FakeTable:
    def __init__(self, payload):
        self._payload = payload

    def to_pydict(self):
        return self._payload


def test_load_all_frame_state_rows_reads_each_data_file_once(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    meta_file = dataset_root / "meta" / "episodes" / "chunk-000" / "episodes.parquet"
    data_file = dataset_root / "data" / "chunk-000" / "file-000000.parquet"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.touch()
    data_file.touch()

    read_calls: list[tuple[str, tuple[str, ...] | None]] = []

    def fake_read_table(path: str, columns=None):
        read_calls.append((path, tuple(columns) if columns is not None else None))
        if path == str(meta_file):
            return _FakeTable(
                {
                    "episode_index": [7, 8],
                    "data/chunk_index": [0, 0],
                    "data/file_index": [0, 0],
                }
            )
        if path == str(data_file):
            return _FakeTable(
                {
                    "episode_index": [7, 7, 8, 8],
                    "observation.state": [
                        [1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0],
                        [4.0, 40.0],
                    ],
                }
            )
        raise AssertionError(f"unexpected parquet path: {path}")

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.read_table = fake_read_table
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    monkeypatch.setattr(compare_runtime.infer_runtime, "_resolve_repo_path", lambda path: Path(path))

    frame_states, frame_refs = compare_runtime._load_all_frame_state_rows(dataset_root, [7, 8])

    assert frame_refs == [(7, 0), (7, 1), (8, 0), (8, 1)]
    assert np.allclose(frame_states, [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    assert read_calls.count((str(data_file), ("episode_index", "observation.state"))) == 1
