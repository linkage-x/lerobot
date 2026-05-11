#!/usr/bin/env python

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.handheld import handheld_soft_sync


def test_build_report_uses_raw_capture_timestamps(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.device_capture_timestamp": {
                        "dtype": "float64",
                        "shape": [2],
                        "names": [
                            "camera.front.capture_timestamp_s",
                            "handheld_gripper.pika.capture_timestamp_s",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "handheld_raw_capture.json").write_text(
        json.dumps({"soft_sync_applied": False}),
        encoding="utf-8",
    )
    table = pa.table(
        {
            "timestamp": [0.0, 1.0 / 30.0],
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "observation.device_capture_timestamp": [
                [0.001, 0.004],
                [0.030, 0.070],
            ],
        }
    )
    pq.write_table(table, dataset_root / "data" / "chunk-000" / "file-000.parquet")

    report = handheld_soft_sync.build_report(
        dataset_root=dataset_root,
        tolerance_ms=20.0,
        global_lag_tolerance_ms=50.0,
    )

    assert report["total_frames"] == 2
    assert report["raw_capture_metadata_present"] is True
    assert report["raw_capture_soft_sync_applied"] is False
    assert report["summary"]["skew_over_tolerance_frames"] == 1
    assert report["summary"]["nonfinite_capture_timestamp_frames"] == 0
    assert report["summary"]["max_skew_s"]["max"] == pytest.approx(0.04)
