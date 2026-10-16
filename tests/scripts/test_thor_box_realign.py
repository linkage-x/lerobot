"""box_realign: a tracked dataset's BOX columns re-paired at mid-exposure + each BOX's delay."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from tools.thor.gmsl2 import (
    box_realign as br,  # noqa: E402
    thor_lerobot_v3 as lr3,  # noqa: E402
)

FPS = 60
FRAMES = 30
T0_WALL = 1_791_966_174.0
T0_MONO = 712_506.0
SOF0 = 0.2  # first frame's SOF, t0-relative
EXPOSURE_S = 0.010
DELAY_S = {"boxA": 0.009, "boxB": 0.0}  # how late each BOX's stamps really are


def _sof(f: int) -> float:
    return SOF0 + f / FPS


def _make_dataset(tmp_path):
    """A one-episode recording paired the way the recorder did before 2026-10-15 (raw SOF, nearest)."""
    root = tmp_path / "thor_gmsl2_10ch_v1_20261015_120000"
    ep = root / "episodes" / "episode_000000"
    ep.mkdir(parents=True)
    rows = ["logical_frame_index,sensor_timestamp_ns,sensor_exposure_time_ns"]
    rows += [
        f"{f},{int(round((T0_MONO + _sof(f)) * 1e9))},{int(EXPOSURE_S * 1e9)}" for f in range(FRAMES + 5)
    ]
    (ep / "cam_03.argus_frame_metadata.csv").write_text("\n".join(rows) + "\n")
    samples: dict[str, list] = {}
    lines = []
    for box, delay in DELAY_S.items():
        for i in range(1000):  # 1 kHz for 1 s: the gripper reading is the true sample time
            t_true = i / 1000.0
            wall = T0_WALL + t_true + delay
            d = {"timestamp": 1_000_000 + i * 1000, "distance_m": t_true}
            samples.setdefault(f"{box}/box_gripper", []).append(
                {"t_rel_s": wall - T0_WALL, "wall_s": wall, "data": d}
            )
            lines.append(
                json.dumps(
                    {
                        "sid": f"{box}/box_gripper",
                        "mcu_ts": d["timestamp"],
                        "wall_s": wall,
                        "t_rel_s": wall - T0_WALL,
                        "data": d,
                    }
                )
            )
    (ep / "box_sensors.jsonl").write_text("\n".join(lines) + "\n")
    (ep / "meta.json").write_text(
        json.dumps(
            {
                "sync_reference": {"t0_wall_s": T0_WALL, "t0_mono_s": T0_MONO},
                "box_camera_alignment": {"mode": "sensor_timestamp_sof", "exposure_fraction": 0.0},
            }
        )
    )
    sof = lr3.camera_frame_times_rel(ep, T0_MONO, exposure_fraction=0.0)
    lr3.write_box_lerobot_v3_episode(
        root,
        repo_id="x",
        task="t",
        fps=FPS,
        episode_index=0,
        snapshots=[],
        duration_s=FRAMES / FPS,
        sensor_samples=samples,
        t0_wall_s=T0_WALL,
        frame_times_s=sof,
    )
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["fps"] = FPS
    info_path.write_text(json.dumps(info))
    return root


def _gripper(root, box_slot):
    import pyarrow.parquet as pq

    state = np.asarray(
        pq.read_table(root / "data/chunk-000/file-000.parquet")["observation.state"].to_pylist()
    )
    return state[:, box_slot * len(lr3.BOX_STATE_NAMES)]


def test_realign_reads_a_measured_box_at_mid_exposure_and_leaves_the_other_on_sof(tmp_path):
    root = _make_dataset(tmp_path)
    before = _gripper(root, 0)
    # the recorder's pairing: boxA read at SOF, i.e. 9 ms of stamp delay off the scene (mid-exposure)
    mid = np.array([_sof(f) - EXPOSURE_S / 2 for f in range(FRAMES)])
    assert np.abs(before - (np.array([_sof(f) for f in range(FRAMES)]) - DELAY_S["boxA"])).max() < 1e-3

    report = br.realign(root, estimate=False, write=True, table={"boxA": DELAY_S["boxA"]})
    entry = report["episodes"]["0"]
    assert entry["reproduced_same_pick"] == 1.0
    assert entry["boxes"]["boxA"] == {"target": "mid_exposure_plus_d_box", "source": "table", "d_box_ms": 9.0}
    assert entry["boxes"]["boxB"]["target"] == "raw_sof"
    after_a, after_b = _gripper(root, 0), _gripper(root, 1)
    assert np.abs(after_a - mid).max() < 1e-5  # exactly the scene instant (interpolated, float32)
    assert np.abs(after_b - np.array([_sof(f) for f in range(FRAMES)])).max() < 1e-3  # as before

    # kept beside, recorded, stats follow
    assert (root / "data/chunk-000/file-000.parquet.pre_realign").exists()
    info = json.loads((root / "meta/info.json").read_text())
    assert info["box_alignment"]["episodes"]["0"]["boxes"]["boxA"]["source"] == "table"
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["observation.state"]["max"][0] == pytest.approx(float(after_a.max()), abs=1e-6)

    # a second run (the dataset tracked again) reproduces the first realign and changes nothing
    again = br.realign(root, estimate=False, write=True, table={"boxA": DELAY_S["boxA"]})
    assert (
        again["episodes"]["0"]["reproduced_same_pick"] == 1.0 and again["episodes"]["0"]["max_change"] < 1e-6
    )


def test_an_episode_estimate_wins_over_the_table_only_when_it_correlates(tmp_path, monkeypatch):
    root = _make_dataset(tmp_path)
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": FPS, "derived": {br.SIDECAR: {"summary_json": ""}}})
    )
    monkeypatch.setattr(br, "_tracking_csv", lambda dataset: root / "x.csv")
    monkeypatch.setattr(
        br,
        "estimate_delays",
        lambda dataset, csv: {0: {"boxA": {"d_box_ms": 8.0, "r": 0.95}, "boxB": {"d_box_ms": 3.0, "r": 0.4}}},
    )
    report = br.realign(root, estimate=True, write=False, table={"boxA": 0.009})
    boxes = report["episodes"]["0"]["boxes"]
    assert boxes["boxA"] == {
        "target": "mid_exposure_plus_d_box",
        "source": "episode",
        "d_box_ms": 8.0,
        "r": 0.95,
    }
    assert boxes["boxB"]["target"] == "raw_sof" and boxes["boxB"]["episode_estimate"]["r"] == 0.4
    assert report["estimate"] == "ok"
    assert not (root / "data/chunk-000/file-000.parquet.pre_realign").exists()  # dry run


def test_a_parquet_the_rebuild_cannot_reproduce_is_left_alone(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = _make_dataset(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    ts = np.asarray(table["box.timestamps"].to_pylist()) + 5000.0  # every pick one sample off
    table = table.set_column(
        table.schema.get_field_index("box.timestamps"),
        "box.timestamps",
        pa.array(ts.tolist(), type=table.schema.field("box.timestamps").type),
    )
    pq.write_table(table, path)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        br.realign(root, estimate=False, write=True, table={"boxA": 0.009})
    assert not path.with_name(path.name + ".pre_realign").exists()
