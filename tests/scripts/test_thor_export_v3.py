from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.thor.gmsl2 import export_v3


# ------------------------------------------------------- discovery / order ---


def _make_session(root: Path, name: str, episodes: list[int], cams=("cam_00",)) -> Path:
    session = root / name
    for ep in episodes:
        ep_dir = session / "episodes" / f"episode_{ep:06d}"
        ep_dir.mkdir(parents=True)
        (ep_dir / "meta.json").write_text(
            json.dumps(
                {
                    "video": {"fps": 30, "width": 16, "height": 16, "replay_warmup_s": 0.0},
                    "cameras": [{"name": c, "file": f"{c}.mkv"} for c in cams],
                }
            ),
            encoding="utf-8",
        )
        for c in cams:
            (ep_dir / f"{c}.mkv").write_bytes(b"0")
    return session


def test_find_task_sessions_strips_namespace_and_sorts_by_timestamp(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _make_session(datasets, "pick_and_place_20260601_101320", [0])
    _make_session(datasets, "pick_and_place_20260601_101046", [0])
    _make_session(datasets, "pick_and_place", [0])
    _make_session(datasets, "fold_towel", [0])

    found = export_v3.find_task_sessions(datasets, "local/pick_and_place")

    assert [p.name for p in found] == [
        "pick_and_place",
        "pick_and_place_20260601_101046",
        "pick_and_place_20260601_101320",
    ]


def test_gather_episodes_preserves_session_then_episode_order(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    _make_session(datasets, "pick_and_place_20260601_101046", [0, 1])
    _make_session(datasets, "pick_and_place_20260601_101320", [0])
    sessions = export_v3.find_task_sessions(datasets, "pick_and_place")

    episodes = export_v3.gather_episodes(sessions)

    assert [(e.session_dir.name, e.local_index) for e in episodes] == [
        ("pick_and_place_20260601_101046", 0),
        ("pick_and_place_20260601_101046", 1),
        ("pick_and_place_20260601_101320", 0),
    ]


# ------------------------------------------------------------- end-to-end ----


def _write_mkv(path: Path, n_frames: int, w: int, h: int, fps: int) -> None:
    import av
    import numpy as np

    container = av.open(str(path), mode="w", format="matroska")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        img = np.full((h, w, 3), (i * 20) % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_box_parquet(session: Path, ep_frames: dict[int, int], state_width: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = session / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    rows = {k: [] for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")}
    idx = 0
    for ep, n in ep_frames.items():
        for f in range(n):
            rows["observation.state"].append([float(f)] * state_width)
            rows["action"].append([float(f)] * state_width)
            rows["timestamp"].append(float(f) / 30.0)
            rows["frame_index"].append(f)
            rows["episode_index"].append(ep)
            rows["index"].append(idx)
            rows["task_index"].append(0)
            idx += 1
    schema = pa.schema(
        [
            ("observation.state", pa.list_(pa.float32(), state_width)),
            ("action", pa.list_(pa.float32(), state_width)),
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ]
    )
    pq.write_table(pa.table(rows, schema=schema), data_dir / "file-000.parquet")


def _make_video_session(root: Path, name: str, ep_frames: dict[int, int], *, cams, w, h, fps, with_box):
    session = root / name
    for ep, n in ep_frames.items():
        ep_dir = session / "episodes" / f"episode_{ep:06d}"
        ep_dir.mkdir(parents=True)
        (ep_dir / "meta.json").write_text(
            json.dumps(
                {
                    "duration_s": n / fps,
                    "video": {"fps": fps, "width": w, "height": h, "codec": "h264", "replay_warmup_s": 0.0},
                    "cameras": [{"name": c, "file": f"{c}.mkv"} for c in cams],
                }
            ),
            encoding="utf-8",
        )
        for c in cams:
            _write_mkv(ep_dir / f"{c}.mkv", n, w, h, fps)
    if with_box:
        _write_box_parquet(session, ep_frames, state_width=4)
    return session


@pytest.mark.parametrize("with_box", [True, False])
def test_export_task_to_v3_produces_loadable_dataset(tmp_path, with_box):
    import shutil as _sh

    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    if _sh.which("ffmpeg") is None and _sh.which("gst-launch-1.0") is None:
        pytest.skip("no transcoder (ffmpeg / gst-launch-1.0) available")
    LeRobotDataset = pytest.importorskip("lerobot.datasets.lerobot_dataset").LeRobotDataset

    datasets = tmp_path / "datasets"
    exports = tmp_path / "exports"
    datasets.mkdir()
    cams = ("cam_00", "cam_01")
    _make_video_session(datasets, "pick_and_place_20260601_101046", {0: 5}, cams=cams, w=64, h=64, fps=30, with_box=with_box)
    _make_video_session(datasets, "pick_and_place_20260601_101320", {0: 4}, cams=cams, w=64, h=64, fps=30, with_box=with_box)

    out = export_v3.export_task_to_v3(
        datasets_root=datasets,
        exports_root=exports,
        base_name="local/pick_and_place",
        repo_id="local/pick_and_place",
        task="pick the cube",
    )

    assert out == exports / "pick_and_place"
    # Per-episode H.264 mp4 files in the v3 video layout.
    assert (out / "videos" / "observation.images.cam_00" / "chunk-000" / "file-000.mp4").is_file()
    assert (out / "videos" / "observation.images.cam_00" / "chunk-000" / "file-001.mp4").is_file()

    ds = LeRobotDataset("local/pick_and_place", root=out)
    assert ds.meta.total_episodes == 2
    assert ds.meta.total_frames == 9  # 5 + 4
    assert set(ds.meta.video_keys) == {"observation.images.cam_00", "observation.images.cam_01"}
    if with_box:
        assert "observation.state" in ds.meta.features
    else:
        assert "observation.state" not in ds.meta.features

    # Real proof of loadability: a frame must decode to the declared shape.
    item = ds[0]
    assert tuple(item["observation.images.cam_00"].shape)[-2:] == (64, 64)

    sources = json.loads((out / "meta" / "export_sources.json").read_text())
    assert [e["frames"] for e in sources["episodes"]] == [5, 4]
    assert [e["session"] for e in sources["episodes"]] == [
        "pick_and_place_20260601_101046",
        "pick_and_place_20260601_101320",
    ]


def test_export_refuses_existing_output_without_overwrite(tmp_path):
    pytest.importorskip("av")
    datasets = tmp_path / "datasets"
    exports = tmp_path / "exports"
    datasets.mkdir()
    _make_video_session(datasets, "pick_and_place_20260601_101046", {0: 3}, cams=("cam_00",), w=64, h=64, fps=30, with_box=False)
    (exports / "pick_and_place").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="already exists"):
        export_v3.export_task_to_v3(
            datasets_root=datasets,
            exports_root=exports,
            base_name="pick_and_place",
            repo_id="local/pick_and_place",
            task="t",
        )


def test_export_raises_when_no_sessions(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    with pytest.raises(RuntimeError, match="No recorded sessions"):
        export_v3.export_task_to_v3(
            datasets_root=datasets,
            exports_root=tmp_path / "exports",
            base_name="missing",
            repo_id="local/missing",
            task="t",
        )


def test_load_box_rows_reads_box_timestamps(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    session = tmp_path / "sess_20260615_120000"
    data_dir = session / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32(), 2)),
        ("action", pa.list_(pa.float32(), 2)),
        ("box.timestamps", pa.list_(pa.float64(), 2)),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    tbl = pa.table({
        "observation.state": [[1.0, 2.0]],
        "action": [[1.0, 2.0]],
        "box.timestamps": [[1.0e8, 3.5]],
        "timestamp": [0.0],
        "frame_index": [0],
        "episode_index": [0],
        "index": [0],
        "task_index": [0],
    }, schema=schema)
    pq.write_table(tbl, data_dir / "file-000.parquet")

    rows = export_v3._load_box_rows(session)
    assert rows[0][0]["box.timestamps"] == [1.0e8, 3.5]


def test_v3writer_carries_box_timestamps_through(tmp_path):
    pa = pytest.importorskip("pyarrow")  # noqa: F841
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "out"
    writer = export_v3._V3Writer(
        out, repo_id="x/y", task="t", fps=30, height=0, width=0,
        video_keys=[], state_width=4, state_names=None,
        ts_width=3, ts_names=["a.timestamp", "b.timestamp", "received_wall_time_s"],
    )
    writer.append_episode(
        episode_index=0, n_frames=2,
        state_rows=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        action_rows=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        video_files={},
        ts_rows=[[1.0e8, 2.0e8, 1.5], [1.0e8 + 5, 2.0e8 + 5, 2.5]],
    )
    writer.finalize()

    table = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    assert "box.timestamps" in table.column_names
    assert table.column("box.timestamps").to_pylist()[0] == [1.0e8, 2.0e8, 1.5]
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["features"]["box.timestamps"]["dtype"] == "float64"
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert "box.timestamps" in stats
