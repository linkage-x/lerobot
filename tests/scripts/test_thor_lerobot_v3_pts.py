import json
from pathlib import Path

import pytest

from tools.thor.gmsl2 import thor_lerobot_v3 as lr3


def test_parse_ffprobe_pts_ignores_non_float_lines():
    assert lr3._parse_ffprobe_pts("0.100000\nN/A\n0.116667\n") == [0.1, 0.116667]


def test_extract_pts_uses_ffprobe_result(monkeypatch, tmp_path):
    calls = []

    def fake_ffprobe(path: Path, *, timeout_s: float):
        calls.append((path, timeout_s))
        return [1.0, 1.5]

    def fake_gstreamer(path: Path, *, timeout_s: float):
        raise AssertionError(
            "GStreamer fallback should not run when ffprobe returns a result"
        )

    monkeypatch.setattr(lr3, "_extract_pts_ffprobe", fake_ffprobe)
    monkeypatch.setattr(lr3, "_extract_pts_gstreamer", fake_gstreamer)

    mkv = tmp_path / "cam_00.mkv"
    assert lr3.extract_pts(mkv, timeout_s=2.0) == [1.0, 1.5]
    assert calls == [(mkv, 2.0)]


def test_extract_pts_falls_back_to_gstreamer_when_ffprobe_missing(monkeypatch, tmp_path):
    calls = []

    def fake_ffprobe(path: Path, *, timeout_s: float):
        calls.append(("ffprobe", path, timeout_s))
        return None

    def fake_gstreamer(path: Path, *, timeout_s: float):
        calls.append(("gstreamer", path, timeout_s))
        return [1.601, 1.655]

    monkeypatch.setattr(lr3, "_extract_pts_ffprobe", fake_ffprobe)
    monkeypatch.setattr(lr3, "_extract_pts_gstreamer", fake_gstreamer)

    mkv = tmp_path / "cam_00.mkv"
    assert lr3.extract_pts(mkv, timeout_s=3.0) == [1.601, 1.655]
    assert calls == [("ffprobe", mkv, 3.0), ("gstreamer", mkv, 3.0)]


def _minimal_snapshot(distance_m: float, t_relative_s: float = 0.0) -> dict:
    return {
        "valid": True,
        "t_relative_s": t_relative_s,
        "sensors": {
            "box_gripper": {
                "distance_m": distance_m,
                "timestamp": t_relative_s,
            }
        },
    }


def test_lr3_writer_appends_without_reading_existing_parquet(monkeypatch, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    real_read_table = pq.read_table
    read_calls = []

    def spy_read_table(where, *args, **kwargs):
        read_calls.append(Path(where))
        return real_read_table(where, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy_read_table)

    writer = lr3.Lr3Writer(tmp_path, repo_id="repo", task="pick", fps=2)
    data_path = writer.append_episode(
        episode_index=0,
        snapshots=[_minimal_snapshot(0.10)],
    )
    writer.append_episode(
        episode_index=1,
        snapshots=[_minimal_snapshot(0.20)],
    )

    assert read_calls == []

    writer.finalize()
    assert read_calls == [data_path]

    table = real_read_table(data_path)
    assert table.num_rows == 2
    assert table["episode_index"].to_pylist() == [0, 1]
    assert table["frame_index"].to_pylist() == [0, 0]
    assert table["index"].to_pylist() == [0, 1]

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 2
    assert (tmp_path / "meta" / "stats.json").exists()

    episodes = real_read_table(
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert episodes["episode_index"].to_pylist() == [0, 1]
    assert episodes["dataset_from_index"].to_pylist() == [0, 1]
    assert episodes["dataset_to_index"].to_pylist() == [1, 2]
