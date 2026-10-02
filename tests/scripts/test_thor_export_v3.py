from __future__ import annotations

import json
import math
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
        _write_online_sync_manifest(ep_dir, cams, n_frames=1)
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


def _write_box_parquet(
    session: Path,
    ep_frames: dict[int, int],
    state_width: int,
    *,
    include_action: bool = True,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = session / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    keys = ["observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    if include_action:
        keys.insert(1, "action")
    rows = {k: [] for k in keys}
    idx = 0
    for ep, n in ep_frames.items():
        for f in range(n):
            rows["observation.state"].append([float(f)] * state_width)
            if include_action:
                rows["action"].append([float(f)] * state_width)
            rows["timestamp"].append(float(f) / 30.0)
            rows["frame_index"].append(f)
            rows["episode_index"].append(ep)
            rows["index"].append(idx)
            rows["task_index"].append(0)
            idx += 1
    fields = [("observation.state", pa.list_(pa.float32(), state_width))]
    if include_action:
        fields.append(("action", pa.list_(pa.float32(), state_width)))
    fields.extend([
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    schema = pa.schema(fields)
    pq.write_table(pa.table(rows, schema=schema), data_dir / "file-000.parquet")


def _write_online_sync_manifest(ep_dir: Path, cams: tuple[str, ...] | list[str], n_frames: int, *, ok: bool = True) -> None:
    (ep_dir / "online_sync_manifest.json").write_text(
        json.dumps(
            {
                "ok": ok,
                "failure": "" if ok else "test failure",
                "fps": 30,
                "target_frames": 0,
                "actual_frames": n_frames,
                "sync_source": "sof_tsc_ns",
                "tolerance_ns": 1_000_000,
                "frame_count_by_camera": {camera: n_frames for camera in cams},
                "max_abs_delta_ns_by_camera": {camera: 12_000 for camera in cams},
                "active_cameras": list(cams),
            }
        ),
        encoding="utf-8",
    )


def _make_video_session(
    root: Path, name: str, ep_frames: dict[int, int], *, cams, w, h, fps, with_box,
    world_frame_id: str | None = None,
):
    session = root / name
    for ep, n in ep_frames.items():
        ep_dir = session / "episodes" / f"episode_{ep:06d}"
        ep_dir.mkdir(parents=True)
        meta = {
            "duration_s": n / fps,
            "video": {"fps": fps, "width": w, "height": h, "codec": "h264", "replay_warmup_s": 0.0},
            "cameras": [{"name": c, "file": f"{c}.mkv"} for c in cams],
        }
        if world_frame_id is not None:
            meta["world_frame"] = {"world_frame_id": world_frame_id, "status": "ok"}
        (ep_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for c in cams:
            _write_mkv(ep_dir / f"{c}.mkv", n, w, h, fps)
        _write_online_sync_manifest(ep_dir, cams, n_frames=n)
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
    _make_video_session(datasets, "pick_and_place_20260601_101046", {0: 5}, cams=cams, w=64, h=64, fps=30, with_box=with_box, world_frame_id="world_20260819_031843")
    _make_video_session(datasets, "pick_and_place_20260601_101320", {0: 4}, cams=cams, w=64, h=64, fps=30, with_box=with_box, world_frame_id="world_20260819_031843")

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

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["world_frame"]["world_frame_id"] == "world_20260819_031843"

    sources = json.loads((out / "meta" / "export_sources.json").read_text())
    assert [e["world_frame_id"] for e in sources["episodes"]] == [
        "world_20260819_031843",
        "world_20260819_031843",
    ]
    assert [e["frames"] for e in sources["episodes"]] == [5, 4]
    assert [e["session"] for e in sources["episodes"]] == [
        "pick_and_place_20260601_101046",
        "pick_and_place_20260601_101320",
    ]


def test_export_task_to_v3_can_write_to_distinct_output_name(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    exports = tmp_path / "exports"
    datasets.mkdir()
    _make_session(datasets, "thor_gmsl2_11ch_v1_20260713_075106", [0], cams=("cam_00", "cam_06"))
    _write_online_sync_manifest(
        datasets / "thor_gmsl2_11ch_v1_20260713_075106" / "episodes" / "episode_000000",
        ("cam_00", "cam_06"),
        n_frames=1,
    )
    monkeypatch.setattr(export_v3, "_mkv_frame_count", lambda _path: 1)

    def fake_transcode(_src, dst, _codec, _fps):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"mp4")

    monkeypatch.setattr(export_v3, "transcode_to_h264_mp4", fake_transcode)

    out = export_v3.export_task_to_v3(
        datasets_root=datasets,
        exports_root=exports,
        base_name="thor_gmsl2_11ch_v1_20260713_075106",
        output_name="thor_gmsl2_2ch_v1_20260713_075106",
        repo_id="local/thor_gmsl2_2ch_v1_20260713_075106",
        task="t",
    )

    assert out == exports / "thor_gmsl2_2ch_v1_20260713_075106"
    sources = json.loads((out / "meta" / "export_sources.json").read_text())
    assert sources["base_name"] == "thor_gmsl2_11ch_v1_20260713_075106"
    assert sources["output_name"] == "thor_gmsl2_2ch_v1_20260713_075106"
    assert sources["repo_id"] == "local/thor_gmsl2_2ch_v1_20260713_075106"


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


def test_export_task_to_v3_merges_tracking_pose_sidecars(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("pyarrow")

    datasets = tmp_path / "datasets"
    exports = tmp_path / "exports"
    datasets.mkdir()
    session = _make_session(datasets, "pick_and_place_20260601_101046", [0], cams=("cam_00",))
    _write_box_parquet(session, {0: 2}, state_width=4)
    _write_online_sync_manifest(session / "episodes" / "episode_000000", ("cam_00",), n_frames=2)

    sidecar = session / "derived" / "april_cube_tracking_in_robot_base"
    sidecar.mkdir(parents=True)
    (sidecar / "state_action.left.csv").write_text(
        """episode_index,frame_index,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw,action_x_m,action_y_m,action_z_m,action_qx,action_qy,action_qz,action_qw
0,0,0.1,0.2,0.3,0,0,0,1,1.1,1.2,1.3,0,0,0,1
0,1,0.4,0.5,0.6,0,0,0,1,1.4,1.5,1.6,0,0,0,1
0,2,9.0,9.0,9.0,0,0,0,1,9.0,9.0,9.0,0,0,0,1
""",
        encoding="utf-8",
    )
    (sidecar / "cube_pose.left.cam_00.csv").write_text(
        """episode_index,frame_index,cube_name,stream_key,cube_cam_x_m,cube_cam_y_m,cube_cam_z_m,cube_cam_qx,cube_cam_qy,cube_cam_qz,cube_cam_qw
0,0,left,cam_00,2.1,2.2,2.3,0,0,0,1
0,1,left,cam_00,2.4,2.5,2.6,0,0,0,1
""",
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    tracking = repo_root / "outputs" / "tracking_analysis" / f"{session.name}_thor_april_tracking_in_robot_base"
    tracking.mkdir(parents=True)
    (tracking / "fused_ee_pose_in_robot_base_records_left.csv").write_text(
        """episode_index,frame_index,cube_name,cube_base_x_m,cube_base_y_m,cube_base_z_m,cube_base_qx,cube_base_qy,cube_base_qz,cube_base_qw
0,0,left,3.1,3.2,3.3,0,0,0,1
0,1,left,3.4,3.5,3.6,0,0,0,1
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(export_v3, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(export_v3, "_mkv_frame_count", lambda _path: 2)

    def fake_transcode(_src, dst, _codec, _fps):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"mp4")

    monkeypatch.setattr(export_v3, "transcode_to_h264_mp4", fake_transcode)

    out = export_v3.export_task_to_v3(
        datasets_root=datasets,
        exports_root=exports,
        base_name="pick_and_place",
        repo_id="local/pick_and_place",
        task="pick the cube",
    )

    table = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    assert "observation.ee_pose.left.base" in table.column_names
    assert "action.ee_pose.left.base" in table.column_names
    assert "observation.cube_pose.left.camera.cam_00" in table.column_names
    assert "observation.cube_pose.left.base" in table.column_names
    ee_rows = table.column("observation.ee_pose.left.base").to_pylist()
    assert ee_rows[0] == pytest.approx([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert ee_rows[1] == pytest.approx([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0])
    assert table.column("action.ee_pose.left.base").to_pylist()[1][:3] == pytest.approx([1.4, 1.5, 1.6])
    assert table.column("observation.cube_pose.left.camera.cam_00").to_pylist()[0][:3] == pytest.approx([2.1, 2.2, 2.3])
    assert table.column("observation.cube_pose.left.base").to_pylist()[1][:3] == pytest.approx([3.4, 3.5, 3.6])

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["features"]["observation.cube_pose.left.base"]["names"] == list(export_v3._POSE7_FEATURE_NAMES)
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert stats["observation.ee_pose.left.base"]["count"] == [2] * 7
    assert not math.isnan(table.column("observation.ee_pose.left.base").to_pylist()[1][0])


def test_export_task_to_v3_falls_back_to_sidecar_cube_base(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("pyarrow")

    datasets = tmp_path / "datasets"
    exports = tmp_path / "exports"
    datasets.mkdir()
    session = _make_session(datasets, "pick_and_place_20260601_101046", [0], cams=("cam_00",))
    _write_box_parquet(session, {0: 2}, state_width=4)
    _write_online_sync_manifest(session / "episodes" / "episode_000000", ("cam_00",), n_frames=2)

    sidecar = session / "derived" / "april_cube_tracking_in_robot_base"
    sidecar.mkdir(parents=True)
    (sidecar / "state_action.left.csv").write_text(
        """episode_index,frame_index,state_x_m,state_y_m,state_z_m,state_qx,state_qy,state_qz,state_qw,action_x_m,action_y_m,action_z_m,action_qx,action_qy,action_qz,action_qw
0,0,0.1,0.2,0.3,0,0,0,1,1.1,1.2,1.3,0,0,0,1
0,1,0.4,0.5,0.6,0,0,0,1,1.4,1.5,1.6,0,0,0,1
""",
        encoding="utf-8",
    )
    (sidecar / "cube_pose.left.cam_00.csv").write_text(
        """episode_index,frame_index,cube_name,stream_key,used_for_fusion,cube_cam_x_m,cube_cam_y_m,cube_cam_z_m,cube_cam_qx,cube_cam_qy,cube_cam_qz,cube_cam_qw,cube_base_x_m,cube_base_y_m,cube_base_z_m,cube_base_qx,cube_base_qy,cube_base_qz,cube_base_qw
0,0,left,cam_00,0,2.1,2.2,2.3,0,0,0,1,3.1,3.2,3.3,0,0,0,1
0,1,left,cam_00,1,2.4,2.5,2.6,0,0,0,1,3.4,3.5,3.6,0,0,0,1
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(export_v3, "_REPO_ROOT", tmp_path / "repo_without_tracking_analysis")
    monkeypatch.setattr(export_v3, "_mkv_frame_count", lambda _path: 2)

    def fake_transcode(_src, dst, _codec, _fps):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"mp4")

    monkeypatch.setattr(export_v3, "transcode_to_h264_mp4", fake_transcode)

    out = export_v3.export_task_to_v3(
        datasets_root=datasets,
        exports_root=exports,
        base_name="pick_and_place",
        repo_id="local/pick_and_place",
        task="pick the cube",
    )

    table = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    assert "observation.cube_pose.left.base" in table.column_names
    assert table.column("observation.cube_pose.left.base").to_pylist()[0][:3] == pytest.approx([3.1, 3.2, 3.3])
    assert table.column("observation.cube_pose.left.base").to_pylist()[1][:3] == pytest.approx([3.4, 3.5, 3.6])
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert stats["observation.cube_pose.left.base"]["count"] == [2] * 7


# ----------------------------------------------- box ↔ camera time sync ----


def test_online_sync_grid_from_manifest_uses_actual_frames(tmp_path):
    ep_dir = tmp_path / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    _write_online_sync_manifest(ep_dir, ("cam_00", "cam_01"), n_frames=7)

    n_frames, manifest = export_v3._online_sync_grid_from_manifest(ep_dir, ["cam_00", "cam_01"])

    assert n_frames == 7
    assert manifest["sync_source"] == "sof_tsc_ns"


def test_online_sync_grid_from_manifest_rejects_count_mismatch(tmp_path):
    ep_dir = tmp_path / "episodes" / "episode_000000"
    ep_dir.mkdir(parents=True)
    _write_online_sync_manifest(ep_dir, ("cam_00", "cam_01"), n_frames=7)
    manifest = json.loads((ep_dir / "online_sync_manifest.json").read_text())
    manifest["frame_count_by_camera"]["cam_01"] = 6
    (ep_dir / "online_sync_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="frame counts"):
        export_v3._online_sync_grid_from_manifest(ep_dir, ["cam_00", "cam_01"])


def _box_row(frame_index: int, value: float, ts_width: int = 1) -> dict:
    return {
        "frame_index": frame_index,
        "observation.state": [value],
        "action": [value * 10],
        "box.timestamps": [float(frame_index)] * ts_width,
    }


def test_align_box_by_frame_index_drops_phantom_grid_tail():
    # Recorder box grid (round(duration*fps)) is often a few frames longer than
    # the real camera clip; the extra phantom rows must be dropped, not shift
    # the remaining ones.
    box_rows = [_box_row(i, float(i)) for i in range(5)]  # frame_index 0..4
    state, action, ts, missing = export_v3._align_box_rows_by_frame_index(
        box_rows, n_frames=3, state_width=1, ts_width=1
    )
    assert missing == 0
    assert state == [[0.0], [1.0], [2.0]]
    assert action == [[1.0], [2.0], [2.0]]
    assert ts == [[0.0], [1.0], [2.0]]


def test_align_box_by_frame_index_carries_last_when_camera_longer():
    # Camera clip longer than the box grid: trailing frames have an image but no
    # box sample -> hold the last reading and count them.
    box_rows = [_box_row(0, 0.0), _box_row(1, 1.0)]  # frame_index 0,1
    state, action, ts, missing = export_v3._align_box_rows_by_frame_index(
        box_rows, n_frames=4, state_width=1, ts_width=1
    )
    assert missing == 2
    assert state == [[0.0], [1.0], [1.0], [1.0]]  # frames 2,3 hold frame 1
    assert action == [[1.0], [1.0], [1.0], [1.0]]


def test_align_box_by_frame_index_uses_frame_index_not_position():
    # A gap at frame_index 1 (rows are 0 and 2). Positional slicing would put the
    # fi=2 row at output frame 1; frame_index keying must place it at frame 2.
    box_rows = [_box_row(0, 10.0), _box_row(2, 12.0)]
    state, action, _ts, missing = export_v3._align_box_rows_by_frame_index(
        box_rows, n_frames=3, state_width=1, ts_width=1
    )
    assert missing == 1  # frame 1 has no row
    assert state == [[10.0], [10.0], [12.0]]  # frame 1 holds frame 0, frame 2 correct
    assert action == [[10.0], [12.0], [12.0]]




def test_load_box_rows_accepts_recorder_parquet_without_action(tmp_path: Path):
    pq = pytest.importorskip("pyarrow.parquet")
    session = tmp_path / "session"
    _write_box_parquet(session, {0: 3}, state_width=1, include_action=False)

    rows_by_ep = export_v3._load_box_rows(session)
    assert "action" not in pq.read_table(session / "data" / "chunk-000" / "file-000.parquet").column_names
    rows = rows_by_ep[0]
    state, action, _ts, missing = export_v3._align_box_rows_by_frame_index(
        rows, n_frames=3, state_width=1, ts_width=0
    )

    assert missing == 0
    assert state == [[0.0], [1.0], [2.0]]
    assert action == [[1.0], [2.0], [2.0]]


def test_align_box_by_frame_index_no_timestamps_when_ts_width_zero():
    box_rows = [_box_row(0, 0.0), _box_row(1, 1.0)]
    _state, _action, ts, _missing = export_v3._align_box_rows_by_frame_index(
        box_rows, n_frames=2, state_width=1, ts_width=0
    )
    assert ts is None


# ---------------------------------------------------------- touch geometry ---


def _write_touch_jsonl(ep_dir: Path, *, points: int, tagged: bool) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ("box0/box_touch_left", "box0/box_touch_right"):
        data = {"timestamp": 1, "fz_0p1N": [0] * points, "fx_0p1N": [0] * points, "fy_0p1N": [0] * points}
        if tagged:
            data["model"] = "m2020" if points == 9 else "paxini_l5325"
            data["points"] = points
        rows.append({"sid": sid, "t_rel_s": 0.0, "data": data})
    with (ep_dir / "box_sensors.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_detect_touch_width_reads_the_pad_actually_recorded(tmp_path):
    # The parquet touch columns are fixed-size lists, so the export has to know
    # the taxel count before it opens the writer. The BOX SDK's 239-slot array
    # is not that count -- an M2020 pad reports 9.
    m2020_ep = tmp_path / "m2020" / "episodes" / "episode_000000"
    _write_touch_jsonl(m2020_ep, points=9, tagged=True)
    assert export_v3._detect_touch_width([m2020_ep]) == 9

    # Frames archived before the tag existed fall back to the array length.
    legacy_ep = tmp_path / "legacy" / "episodes" / "episode_000000"
    _write_touch_jsonl(legacy_ep, points=239, tagged=False)
    assert export_v3._detect_touch_width([legacy_ep]) == 239

    untagged_m2020 = tmp_path / "untagged" / "episodes" / "episode_000000"
    _write_touch_jsonl(untagged_m2020, points=9, tagged=False)
    assert export_v3._detect_touch_width([untagged_m2020]) == 9

    # No touch archive at all keeps the legacy default rather than failing.
    empty_ep = tmp_path / "empty" / "episodes" / "episode_000000"
    empty_ep.mkdir(parents=True)
    assert export_v3._detect_touch_width([empty_ep]) == export_v3._TOUCH_SAMPLE_WIDTH_DEFAULT


def test_detect_touch_width_reports_a_pad_swap_mid_collection(tmp_path, capsys):
    m2020_ep = tmp_path / "a" / "episodes" / "episode_000000"
    paxini_ep = tmp_path / "b" / "episodes" / "episode_000000"
    _write_touch_jsonl(m2020_ep, points=9, tagged=True)
    _write_touch_jsonl(paxini_ep, points=239, tagged=True)

    # Widest wins so nothing is truncated away, and the mismatch is said out
    # loud instead of silently zero-padding half the set.
    assert export_v3._detect_touch_width([m2020_ep, paxini_ep]) == 239
    out = capsys.readouterr().out
    assert "disagree on touch taxel count" in out
    assert "9 taxels" in out and "239 taxels" in out


def test_align_touch_rows_keeps_the_detected_width(tmp_path):
    ep_dir = tmp_path / "s" / "episodes" / "episode_000000"
    _write_touch_jsonl(ep_dir, points=9, tagged=True)

    rows, saw = export_v3._align_touch_rows(ep_dir, 3, 30, width=9)

    assert saw is True
    for column, _, _ in export_v3._TOUCH_ARRAY_COLUMNS:
        assert len(rows[column]) == 3
        assert all(len(row) == 9 for row in rows[column])


def test_export_refuses_to_mix_two_world_frames(tmp_path):
    # Absolute poses from two coordinate systems concatenated into one training
    # set are wrong in a way no downstream check can see, so the export has to
    # stop before it writes anything.
    pytest.importorskip("pyarrow")
    from tools.thor.gmsl2 import world_provenance as wp

    datasets = tmp_path / "datasets"
    exports = tmp_path / "exports"
    datasets.mkdir()
    cams = ("cam_00",)
    _make_video_session(datasets, "pick_and_place_20260601_101046", {0: 5}, cams=cams, w=64, h=64, fps=30, with_box=False, world_frame_id="world_20260819_031843")
    _make_video_session(datasets, "pick_and_place_20260601_101320", {0: 4}, cams=cams, w=64, h=64, fps=30, with_box=False, world_frame_id="world_20260901_120000")

    with pytest.raises(wp.MixedWorldError) as excinfo:
        export_v3.export_task_to_v3(
            datasets_root=datasets,
            exports_root=exports,
            base_name="local/pick_and_place",
            repo_id="local/pick_and_place",
            task="pick the cube",
        )

    assert "world_20260819_031843" in str(excinfo.value)
    assert "world_20260901_120000" in str(excinfo.value)
    # Nothing written: the refusal must not leave a half-exported dataset.
    assert not (exports / "pick_and_place" / "data").exists()


def test_export_refuses_stamped_mixed_with_legacy_unstamped(tmp_path):
    pytest.importorskip("pyarrow")
    from tools.thor.gmsl2 import world_provenance as wp

    datasets = tmp_path / "datasets"
    exports = tmp_path / "exports"
    datasets.mkdir()
    cams = ("cam_00",)
    _make_video_session(datasets, "pick_and_place_20260601_101046", {0: 5}, cams=cams, w=64, h=64, fps=30, with_box=False, world_frame_id="world_20260819_031843")
    _make_video_session(datasets, "pick_and_place_20260601_101320", {0: 4}, cams=cams, w=64, h=64, fps=30, with_box=False)

    with pytest.raises(wp.MixedWorldError) as excinfo:
        export_v3.export_task_to_v3(
            datasets_root=datasets,
            exports_root=exports,
            base_name="local/pick_and_place",
            repo_id="local/pick_and_place",
            task="pick the cube",
        )

    assert "<unstamped>" in str(excinfo.value)
