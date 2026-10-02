from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.thor import fr3_data
from tools.thor.gmsl2 import export_v3, thor_lerobot_v3 as lr3


def sample(source: float = 100.0, **overrides):
    return {
        "sample_monotonic_s": source,
        "receiver_monotonic_s": source + 0.001,
        "q": list(range(7)), "dq": [0.2] * 7,
        "tau_J": [1.25] * 7, "tau_ext_hat_filtered": [-0.5] * 7,
        "O_T_EE": [float(i) for i in range(16)], "O_F_ext_hat_K": [2.0] * 6,
        "measured_tcp": [0.4, 0.1, 0.5, 0.0, 0.0, 0.2],
        "commanded_ee": [0.41, 0.11, 0.51, 0.01, 0.01, 0.21],
        "gripper_command": 0.6, "control_command_success_rate": 0.999,
        **overrides,
    }


def test_alignment_uses_actual_camera_times_and_nearest_native_sample():
    rows = fr3_data.align_fr3_samples(
        [sample(100.0), sample(100.02, tau_J=[7.0] * 7)],
        [0.004, 0.017, None, 0.10], 100.0, n_frames=5,
    )
    assert rows[0]["fr3.valid"] == [1.0]
    assert rows[1]["observation.fr3.tau_J"] == [7.0] * 7
    assert rows[1]["fr3.timestamps"] == pytest.approx([100.02, 100.021])
    assert rows[0]["observation.fr3.O_T_EE"] == list(range(16))
    assert rows[0]["action"] == pytest.approx([0.41, 0.11, 0.51, 0.01, 0.01, 0.21, 0.6])
    for row in rows[2:]:
        assert row == fr3_data.empty_fr3_row()


@pytest.mark.parametrize("times,origin", [(None, 100.0), ([0.0], None), ([None], 100.0), ([float("nan")], 100.0)])
def test_missing_camera_clock_never_infers_a_valid_fps_grid(times, origin):
    assert fr3_data.align_fr3_samples([sample()], times, origin, n_frames=1) == [fr3_data.empty_fr3_row()]


@pytest.mark.parametrize("override", [
    {"q": [0.0] * 6}, {"tau_J": [float("nan")] * 7},
    {"O_T_EE": None}, {"control_command_success_rate": float("inf")},
])
def test_incomplete_measurements_are_invalid_without_corrupting_valid_actions(override):
    row = fr3_data.align_fr3_samples([sample(**override)], [0.0], 100.0, n_frames=1)[0]
    assert row["fr3.valid"] == [0.0]
    assert row["observation.fr3.q"] == [0.0] * 7
    assert row["fr3.action_valid"] == [1.0]


@pytest.mark.parametrize("override", [{"commanded_ee": [0.0] * 5}, {"gripper_command": None}, {"gripper_command": 1.1}])
def test_action_has_its_own_validity_flag(override):
    row = fr3_data.align_fr3_samples([sample(**override)], [0.0], 100.0, n_frames=1)[0]
    assert row["fr3.valid"] == [1.0]
    assert row["fr3.action_valid"] == [0.0]
    assert row["action"] == [0.0] * 7


def test_nearest_skew_budget_and_no_stale_frame_hold():
    assert fr3_data.align_fr3_samples([sample()], [0.025], 100.0, n_frames=1)[0]["fr3.valid"] == [1.0]
    assert fr3_data.align_fr3_samples([sample()], [0.025001], 100.0, n_frames=1)[0]["fr3.valid"] == [0.0]
    aligned = fr3_data.align_fr3_samples([sample()], [0.0], 100.0, n_frames=1)[0]
    rows = fr3_data.align_fr3_rows_by_frame_index([{**aligned, "frame_index": 1}], 3)
    assert rows[0] == rows[2] == fr3_data.empty_fr3_row()
    assert rows[1]["action"] == aligned["action"]


def test_helper_is_stdlib_only_and_raw_loader_skips_invalid_lines(tmp_path):
    (tmp_path / "fr3_state.jsonl").write_text(json.dumps(sample()) + "\ninvalid\n[]\n", encoding="utf-8")
    assert fr3_data.load_fr3_samples(tmp_path) == [sample()]
    code = "import sys; from tools.thor import fr3_data; assert not any(x in sys.modules for x in ['torch','pyarrow','datasets'])"
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)


def box_snapshots(count=5):
    return [{"t_relative_s": i / 60, "valid": True, "sensors": {
        "box_gripper": {"distance": 0.042, "timestamp": 12345678},
    }} for i in range(count)]


@pytest.mark.parametrize("enabled", [False, True])
def test_live_writer_preserves_box_schema_and_adds_optional_fr3_data(tmp_path, enabled):
    pq = pytest.importorskip("pyarrow.parquet")
    writer = lr3.open_box_lerobot_v3_writer(tmp_path, repo_id="local/fr3", task="test", fps=60, fr3_enabled=enabled)
    writer.append_episode(
        episode_index=0, snapshots=box_snapshots(), n_frames=3,
        frame_times_s=[0.0, 1 / 60, None], t0_mono_s=100.0,
        fr3_samples=[sample(100.0), sample(100.02)],
    )
    writer.finalize()
    table = pq.read_table(tmp_path / "data/chunk-000/file-000.parquet")
    assert table.num_rows == 3
    assert "box.timestamps" in table.column_names
    assert len(table["observation.state"].to_pylist()[0]) == len(lr3.BOX_STATE_NAMES)
    info = json.loads((tmp_path / "meta/info.json").read_text())
    if enabled:
        assert info["robot_type"] == "thor_gmsl2_box_fr3"
        assert info["features"]["action"]["shape"] == [7]
        assert table["observation.fr3.tau_J"].to_pylist()[0] == [1.25] * 7
        assert table["action"].to_pylist()[0] == pytest.approx(sample()["commanded_ee"] + [0.6])
        assert table["fr3.valid"].to_pylist() == [[1.0], [1.0], [0.0]]
        assert table.schema.field("fr3.timestamps").type.value_type == pytest.importorskip("pyarrow").float64()
        stats = json.loads((tmp_path / "meta/stats.json").read_text())
        assert stats["fr3.valid"]["count"] == [3]
    else:
        assert "fr3.valid" not in table.column_names
        assert "action" not in table.column_names
        assert info["robot_type"] == "thor_gmsl2_box"


def test_live_writer_without_camera_times_retains_invalid_robot_rows(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    writer = lr3.Lr3Writer(tmp_path, repo_id="local/fr3", task="test", fps=60, fr3_enabled=True)
    writer.append_episode(episode_index=0, snapshots=[], n_frames=2, fr3_samples=[sample()], t0_mono_s=100.0)
    writer.finalize()
    table = pq.read_table(tmp_path / "data/chunk-000/file-000.parquet")
    assert table["fr3.valid"].to_pylist() == [[0.0], [0.0]]
    assert table["fr3.action_valid"].to_pylist() == [[0.0], [0.0]]


def make_session(root: Path, name: str, *, enabled=True, raw=True, sidecar=True):
    session = root / name
    ep = session / "episodes/episode_000000"
    ep.mkdir(parents=True)
    meta = {"video": {"fps": 60, "height": 16, "width": 16, "codec": "h264"},
            "cameras": [{"name": "cam_07", "file": "cam_07.mkv"}]}
    if enabled:
        meta["fr3_teleop"] = {"enabled": True, "t0_mono_s": 100.0}
    (ep / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (ep / "cam_07.mkv").write_bytes(b"test")
    (ep / "online_sync_manifest.json").write_text(json.dumps({
        "ok": True, "actual_frames": 3, "active_cameras": ["cam_07"],
        "frame_count_by_camera": {"cam_07": 3}, "sync_source": "sof_tsc_ns",
    }), encoding="utf-8")
    if raw:
        (ep / "fr3_state.jsonl").write_text("\n".join(json.dumps(sample(100.0 + i / 60)) for i in range(3)), encoding="utf-8")
    if sidecar:
        (ep / "cam_07.argus_frame_metadata.csv").write_text(
            "logical_frame_index,sensor_timestamp_ns\n" + "".join(f"{i},{int((100 + i / 60) * 1e9)}\n" for i in range(3)), encoding="utf-8",
        )
    return session


def fake_video(monkeypatch):
    monkeypatch.setattr(export_v3, "_mkv_frame_count", lambda _: 3)
    def transcode(_src, dst, _codec, _fps):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"mp4")
    monkeypatch.setattr(export_v3, "transcode_to_h264_mp4", transcode)


@pytest.mark.parametrize("prealigned", [False, True])
def test_export_retains_real_action_torques_and_all_camera_names(tmp_path, monkeypatch, prealigned):
    pq = pytest.importorskip("pyarrow.parquet")
    datasets = tmp_path / "datasets"
    session = make_session(datasets, "fr3_20261002_010000")
    if prealigned:
        writer = lr3.Lr3Writer(session, repo_id="local/fr3", task="test", fps=60, fr3_enabled=True)
        writer.append_episode(episode_index=0, snapshots=box_snapshots(3), n_frames=3,
                              frame_times_s=[0.0, 1 / 60, None], t0_mono_s=100.0,
                              fr3_samples=[sample(100.0), sample(100.02)])
        writer.finalize()
        # The aligned recorder data must win even if raw telemetry changes.
        (session / "episodes/episode_000000/fr3_state.jsonl").write_text(json.dumps(sample(gripper_command=0.1)))
    fake_video(monkeypatch)
    out = export_v3.export_task_to_v3(datasets_root=datasets, exports_root=tmp_path / "exports",
                                      base_name="fr3", repo_id="local/fr3", task="test")
    table = pq.read_table(out / "data/chunk-000/file-000.parquet")
    assert table.num_rows == 3
    assert len(table["action"].to_pylist()[0]) == 7
    assert table["action"].to_pylist()[0] == pytest.approx(sample()["commanded_ee"] + [0.6])
    assert table["observation.fr3.tau_J"].to_pylist()[0] == [1.25] * 7
    assert table["fr3.valid"].to_pylist()[-1] == ([0.0] if prealigned else [1.0])
    info = json.loads((out / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [7]
    assert "observation.images.cam_07" in info["features"]
    assert info["features"]["observation.touch.box_touch_left.fz_0p1N"]["shape"] == [239]


def test_export_missing_hardware_times_does_not_make_valid_fr3_labels(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    datasets = tmp_path / "datasets"
    make_session(datasets, "fr3", sidecar=False)
    fake_video(monkeypatch)
    out = export_v3.export_task_to_v3(datasets_root=datasets, exports_root=tmp_path / "exports",
                                      base_name="fr3", repo_id="local/fr3", task="test")
    table = pq.read_table(out / "data/chunk-000/file-000.parquet")
    assert table["fr3.valid"].to_pylist() == [[0.0]] * 3
    assert table["action"].to_pylist() == [[0.0] * 7] * 3


def test_mixed_export_refusal_preserves_existing_output(tmp_path):
    pytest.importorskip("pyarrow")
    datasets = tmp_path / "datasets"
    make_session(datasets, "fr3_20261002_010000", enabled=True)
    make_session(datasets, "fr3_20261002_020000", enabled=False, raw=False)
    output = tmp_path / "exports/fr3"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("preserved")
    with pytest.raises(RuntimeError, match="Cannot mix FR3-enabled and BOX-only"):
        export_v3.export_task_to_v3(datasets_root=datasets, exports_root=output.parent,
                                   base_name="fr3", repo_id="local/fr3", task="test", overwrite=True)
    assert (output / "keep.txt").read_text() == "preserved"
