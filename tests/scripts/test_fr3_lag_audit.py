#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.fr3 import fr3_lag_audit


def _write_dataset(dataset_root: Path, *, observation_rows: list[list[float]], action_rows: list[list[float]]) -> None:
    (dataset_root / 'meta').mkdir(parents=True)
    (dataset_root / 'data' / 'chunk-000').mkdir(parents=True)
    (dataset_root / 'meta' / 'info.json').write_text(
        json.dumps(
            {
                'fps': 30,
                'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                'features': {
                    'observation.state': {
                        'names': ['ee.x', 'ee.y', 'ee.z', 'ee.qx', 'ee.qy', 'ee.qz', 'ee.qw', 'gripper.pos'],
                    }
                },
            }
        ),
        encoding='utf-8',
    )
    num_rows = len(observation_rows)
    table = pa.table(
        {
            'observation.state': pa.array(observation_rows, type=pa.list_(pa.float32())),
            'action': pa.array(action_rows, type=pa.list_(pa.float32())),
            'episode_index': pa.array([0] * num_rows, type=pa.int64()),
            'frame_index': pa.array(list(range(num_rows)), type=pa.int64()),
        }
    )
    pq.write_table(table, dataset_root / 'data' / 'chunk-000' / 'file-000.parquet')


def test_compute_best_positive_lag_returns_expected_shift():
    observation = [9.0, 8.0, 0.0, 1.0, 2.0, 3.0]
    action = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    lag, mse = fr3_lag_audit.compute_best_positive_lag(
        observation_values=fr3_lag_audit.np.asarray(observation, dtype=fr3_lag_audit.np.float64),
        action_values=fr3_lag_audit.np.asarray(action, dtype=fr3_lag_audit.np.float64),
        episode_index=fr3_lag_audit.np.asarray([0] * len(observation), dtype=fr3_lag_audit.np.int64),
        max_lag=4,
    )

    assert lag == 2
    assert mse == 0.0


def test_main_reports_fixed_lag_and_gripper_transition(tmp_path: Path, capsys):
    dataset_root = tmp_path / 'dataset'
    observation_rows = [
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]
    action_rows = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]
    _write_dataset(dataset_root, observation_rows=observation_rows, action_rows=action_rows)

    exit_code = fr3_lag_audit.main([str(dataset_root)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert 'ee_lag ee.x: frames=2' in captured.out
    assert 'grip_transition: episode=0 frame=1' in captured.out
    assert 'reach_delay_frames=3' in captured.out


def test_main_reports_error_for_corrupted_dataset(tmp_path: Path, capsys):
    dataset_root = tmp_path / 'dataset'
    (dataset_root / 'meta').mkdir(parents=True)
    (dataset_root / 'data' / 'chunk-000').mkdir(parents=True)
    (dataset_root / 'meta' / 'info.json').write_text(
        json.dumps(
            {
                'fps': 30,
                'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                'features': {
                    'observation.state': {
                        'names': ['ee.x', 'ee.y', 'ee.z', 'ee.qx', 'ee.qy', 'ee.qz', 'ee.qw', 'gripper.pos'],
                    }
                },
            }
        ),
        encoding='utf-8',
    )
    (dataset_root / 'data' / 'chunk-000' / 'file-000.parquet').write_bytes(b'PAR1broken')

    exit_code = fr3_lag_audit.main([str(dataset_root)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert 'ERROR:' in captured.out
