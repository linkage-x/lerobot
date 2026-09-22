from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.thor import plot_p0_replay_sync as plot_sync


def test_generate_diagnostics_writes_per_episode_plot_csv_and_summary(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    sidecar_dir = dataset / plot_sync.SIDECAR_RELATIVE_DIR
    parquet_dir = dataset / "data/chunk-000"
    (dataset / "meta").mkdir(parents=True)
    sidecar_dir.mkdir(parents=True)
    parquet_dir.mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {
                        "names": ["unused", plot_sync.GRIPPER_FEATURE_NAME],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0, 0.08], [0.0, 0.04], [0.0, 0.07]],
                "timestamp": [0.0, 0.5, 1.0],
                "episode_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
            }
        ),
        parquet_dir / "file-000.parquet",
    )
    with (sidecar_dir / "state_action.left.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp_s",
                "episode_index",
                "frame_index",
                "state_x_m",
                "state_y_m",
                "state_z_m",
            ],
        )
        writer.writeheader()
        for frame, timestamp in enumerate((0.0, 0.5, 1.0)):
            writer.writerow(
                {
                    "timestamp_s": timestamp,
                    "episode_index": 0,
                    "frame_index": frame,
                    "state_x_m": 0.4 + frame * 0.01,
                    "state_y_m": 0.1,
                    "state_z_m": 0.2,
                }
            )

    output = tmp_path / "diagnostics"
    summary = plot_sync.generate_diagnostics(dataset, output, ("left",))

    episode = summary["sides"]["left"]["episodes"]["0"]
    assert episode["frames"] == 3
    assert episode["first_close_over_0p5mm_s"] == 0.5
    assert episode["first_open_over_0p5mm_s"] == 1.0
    assert Path(episode["plot"]).is_file()
    assert Path(episode["csv"]).is_file()
    assert (output / "summary.json").is_file()
