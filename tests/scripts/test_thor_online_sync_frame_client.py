import json
from pathlib import Path

from tools.thor.gmsl2.online_sync_frame_client import ThorOnlineSyncFrameClient


def _write_bus(root: Path, *, width: int = 4, height: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    size = width * height * 3 // 2
    cameras = {}
    for idx, name in enumerate(("cam_06", "cam_07")):
        frame_path = root / f"slot0_{name}.nv12"
        frame_path.write_bytes(bytes([idx + 1]) * size)
        cameras[name] = {
            "path": str(frame_path),
            "camera": name,
            "logical_frame_index": 42,
            "local_frame_number": 100 + idx,
            "sensor_timestamp_ns": 1000 + idx,
            "sof_tsc_ns": 1_000_000 + idx * 5000,
            "eof_tsc_ns": 1_010_000 + idx * 5000,
            "internal_frame_count": 200 + idx,
        }
    (root / "latest_cluster.json").write_text(json.dumps({
        "version": 1,
        "publish_seq": 7,
        "slot": 0,
        "recording": True,
        "episode_index": 3,
        "logical_frame_index": 42,
        "sync_source": "sof_tsc_ns",
        "format": "nv12",
        "width": width,
        "height": height,
        "min_sof_tsc_ns": 1_000_000,
        "max_sof_tsc_ns": 1_005_000,
        "max_delta_ns": 5000,
        "cameras": cameras,
    }), encoding="utf-8")


def test_frame_client_reads_latest_cluster(tmp_path: Path) -> None:
    _write_bus(tmp_path)
    client = ThorOnlineSyncFrameClient(tmp_path)

    cluster = client.get_latest()

    assert cluster is not None
    assert cluster.publish_seq == 7
    assert cluster.recording is True
    assert cluster.episode_index == 3
    assert cluster.logical_frame_index == 42
    assert cluster.max_delta_ns == 5000
    assert set(cluster.frames) == {"cam_06", "cam_07"}
    assert cluster.frames["cam_06"].read_nv12() == bytes([1]) * 12
    assert cluster.frames["cam_07"].sof_tsc_ns == 1_005_000


def test_frame_client_camera_filter_requires_requested_cameras(tmp_path: Path) -> None:
    _write_bus(tmp_path)

    assert ThorOnlineSyncFrameClient(tmp_path, cameras=["cam_06"]).get_latest() is not None
    assert ThorOnlineSyncFrameClient(tmp_path, cameras=["cam_08"]).get_latest() is None


def test_frame_client_rejects_incomplete_raw_file(tmp_path: Path) -> None:
    _write_bus(tmp_path)
    (tmp_path / "slot0_cam_07.nv12").write_bytes(b"short")

    assert ThorOnlineSyncFrameClient(tmp_path).get_latest() is None
