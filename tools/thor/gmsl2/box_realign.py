#!/usr/bin/env python3
"""
Re-pair a recorded dataset's BOX columns with each BOX's transport delay, once it is tracked.

The recorder reads every BOX at camera mid-exposure + ``d_box`` from the delay table
(``thor_lerobot_v3.EXPOSURE_CENTER_FRACTION`` / ``BOX_TIME_DELAYS_PATH``), and a BOX it has no
delay for at the raw SOF. ``d_box`` moves 1-2 ms between sessions (right box1819152274: 7.4 ms on
10-13, 7.9-9.6 ms on 10-14), and a tracked dataset can measure its own: the gyro against the
tracked cube's angular velocity, per episode (``metrology.cli.estimate_camera_imu_time_offset``).
This tool does that and rebuilds ``observation.state`` / ``box.timestamps`` from the raw episode
data (``box_sensors.jsonl``, the Argus sidecars) with

  * each BOX of each episode at mid-exposure + its own d_box where that episode's estimate is
    trustworthy (peak correlation >= ``MIN_PEAK_CORRELATION``), else the table's, else raw SOF
  * the continuous channels interpolated (``BOX_INTERPOLATED_FIELDS``)

and writes the per-episode choice into ``meta/info.json`` (``box_alignment``). The old parquet and
stats are kept beside them (``*.pre_realign``) the first time.

Before it rewrites anything it rebuilds every episode the way the parquet says it was built
(``meta.json`` ``box_camera_alignment``) and requires that to reproduce the stored columns: a
rebuild that cannot reproduce the old pairing is not trusted to make a new one.

On Thor, from ~/lerobot (needs numpy, scipy, pyarrow -- the opencv_kalibr venv has them):
  PYTHONPATH=.:third_party/opencv_kalibr third_party/opencv_kalibr/.venv/bin/python3 \\
      tools/thor/gmsl2/box_realign.py outputs/datasets/<dataset> --estimate --write
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "opencv_kalibr"))
from tools.thor.gmsl2 import thor_lerobot_v3 as lr3  # noqa: E402

# An episode's own d_box is used only when its gyro and the tracked rotation correlate this well
# (10-14: 0.81-0.98 on episodes with real rotation; a cube barely turned reads 0.4 and is wrong).
MIN_PEAK_CORRELATION = 0.8
# A rebuild of the recorded pairing must pick the same BOX sample (same MCU stamp) for at least
# this share of frames per sensor; the rest sit within float rounding of halfway between two
# samples and may tip to the neighbour (10-14 16:22: 2 of 1203 frames). Where the picks agree the
# stored float32 state must be reproduced to float32 rounding.
REPRODUCE_MIN_SAME_PICK = 0.99
REPRODUCE_TOL = 1e-5
SIDECAR = "april_cube_tracking_in_robot_base"


def load_episode(ep_dir: Path) -> dict[str, Any]:
    """The raw inputs the recorder paired from."""
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    sync = meta.get("sync_reference") or {}
    samples: dict[str, list[dict[str, Any]]] = {}
    with (ep_dir / "box_sensors.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            d = json.loads(line)
            samples.setdefault(d["sid"], []).append(
                {"t_rel_s": d["t_rel_s"], "wall_s": d["wall_s"], "data": d["data"]}
            )
    t0_mono = sync.get("t0_mono_s")
    alignment = meta.get("box_camera_alignment") or {}
    return {
        "meta": meta,
        "samples": samples,
        "t0_wall_s": float(sync.get("t0_wall_s") or 0.0),
        "sof": lr3.camera_frame_times_rel(ep_dir, t0_mono, exposure_fraction=0.0, readout_offset_s=0.0),
        "mid": lr3.camera_frame_times_rel(ep_dir, t0_mono, exposure_fraction=-0.5),
        "recorded_fraction": float(alignment.get("exposure_fraction", 0.0)),
        "recorded_targets": alignment.get("box_targets"),
    }


def build_rows(
    ep: dict[str, Any], n_frames: int, fps: int, box_ids, *, delays_s=None, legacy=False, prior=None
):
    """Rows for one episode: ``legacy`` as the parquet holds it now -- as an earlier realign chose
    (``prior``, its ``boxes`` entry for this episode) or else as the recorder built it (from its
    meta) -- otherwise the new pairing with ``delays_s``."""
    kw = {
        "fps": fps,
        "episode_index": 0,
        "snapshots": [],
        "duration_s": n_frames / fps,
        "sensor_samples": ep["samples"],
        "t0_wall_s": ep["t0_wall_s"],
        "box_ids": box_ids,
    }
    if legacy and prior is not None:
        delays_s = {
            b: c["d_box_ms"] / 1000.0
            for b, c in prior.items()
            if c.get("target") == "mid_exposure_plus_d_box"
        }
        legacy = False
    if legacy:
        if ep["recorded_targets"] is None:  # before 2026-10-15: one timeline, nearest neighbour
            times = ep["sof"] if ep["recorded_fraction"] == 0.0 else ep["mid"]
            return lr3._build_episode_rows(**kw, frame_times_s=times)
        recorded = {
            b: t["d_box_ms"] / 1000.0
            for b, t in ep["recorded_targets"].items()
            if t.get("d_box_ms") is not None
        }
        return lr3._build_episode_rows(
            **kw,
            frame_times_s=ep["mid"],
            sof_frame_times_s=ep["sof"],
            box_delays_s=recorded,
            interpolate=True,
        )
    return lr3._build_episode_rows(
        **kw,
        frame_times_s=ep["mid"],
        sof_frame_times_s=ep["sof"],
        box_delays_s=delays_s or {},
        interpolate=True,
    )


def _reproduces(rows, stored_state, stored_ts, ts_names) -> tuple[float | None, str]:
    """(worst per-sensor share of frames with the same pick, "") if the rebuild reproduces the stored
    columns, else (None, why)."""
    import numpy as np

    if len(rows) != len(stored_state):
        return None, f"{len(rows)} rows rebuilt, {len(stored_state)} stored"
    ts = np.asarray([r["box.timestamps"] for r in rows], dtype=np.float64)
    st = np.asarray([r["observation.state"] for r in rows], dtype=np.float32).astype(np.float64)
    same = ts == stored_ts
    shares = [float(same[:, k].mean()) for k in range(ts.shape[1]) if np.any(stored_ts[:, k])]
    worst_share = min(shares, default=1.0)
    if worst_share < REPRODUCE_MIN_SAME_PICK:
        k = int(
            np.argmin([same[:, k].mean() if np.any(stored_ts[:, k]) else 1.0 for k in range(ts.shape[1])])
        )
        return None, f"{ts_names[k]} picks the stored sample in only {worst_share:.1%} of frames"
    rows_same = same.all(axis=1)
    err = np.abs(st[rows_same] - stored_state[rows_same])
    tol = REPRODUCE_TOL * max(1.0, float(np.max(np.abs(stored_state))))
    if err.size and float(err.max()) > tol:
        return None, f"same samples, different values (worst {float(err.max()):.3g})"
    return round(worst_share, 4), ""


def estimate_delays(dataset: Path, tracking_csv: Path) -> dict[int, dict[str, dict[str, float]]]:
    """``episode -> box -> {d_box_ms, r}`` from the camera<->IMU estimator at mid-exposure."""
    from metrology.cli import estimate_camera_imu_time_offset as est

    with tempfile.TemporaryDirectory() as tmp:
        rc = est.main(
            [
                "--dataset-root", str(dataset),
                "--tracking-csv", str(tracking_csv),
                "--exposure-fraction", "-0.5",
                "--output-dir", tmp,
            ]
        )  # fmt: skip
        path = Path(tmp) / "camera_imu_time_offset.json"
        if not path.is_file():  # no episode the estimator could use: the table stands
            raise RuntimeError(f"the camera<->IMU estimator found nothing to fit (exit {rc})")
        report = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[str, dict[str, float]]] = {}
    for e in report.get("episodes", []):
        if e.get("status") != "ok":
            continue
        split = lr3._split_box_sensor_id(str(e.get("imu_sid", "")))
        if split is None or not split[0]:
            continue  # a one-BOX recording names no device
        r = float(e["coarse"]["peak_normalized_correlation"])
        d = float(e["fine"]["delta_ms"])
        prev = out.setdefault(int(e["episode_index"]), {}).get(split[0])
        if prev is None or r > prev["r"]:  # a BOX seen by two cubes: the better-correlated one
            out[int(e["episode_index"])][split[0]] = {"d_box_ms": round(d, 3), "r": round(r, 3)}
    return out


def _tracking_csv(dataset: Path) -> Path | None:
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    summary = ((info.get("derived") or {}).get(SIDECAR) or {}).get("summary_json")
    if summary and Path(summary).is_file():
        csv_path = Path(summary).parent / "fused_ee_pose_in_robot_base_records.csv"
        if csv_path.is_file():
            return csv_path
    return None


def realign(
    dataset: Path, *, estimate: bool, write: bool, table: dict[str, float] | None = None
) -> dict[str, Any]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    info_path = dataset / "meta" / "info.json"
    data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    if not info_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("no BOX parquet / meta/info.json: nothing recorded to realign")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = int(info.get("fps") or 60)
    table_pq = pq.read_table(data_path)
    episode_col = table_pq["episode_index"].to_numpy()
    state = np.asarray(table_pq["observation.state"].combine_chunks().flatten().to_numpy(), dtype=np.float64)
    width = len(state) // max(table_pq.num_rows, 1)
    state = state.reshape(-1, width)
    ts_width = table_pq.schema.field("box.timestamps").type.list_size
    table = lr3.load_box_time_delays() if table is None else table

    measured: dict[int, dict[str, dict[str, float]]] = {}
    estimate_note = "not asked"
    if estimate:
        csv_path = _tracking_csv(dataset)
        if csv_path is None:
            estimate_note = "not tracked: no fused records to estimate d_box from; the table stands"
        else:
            try:
                measured = estimate_delays(dataset, csv_path)
                estimate_note = "ok"
            except Exception as exc:  # noqa: BLE001 - an episode with no rotation is no reason to stop
                estimate_note = f"{exc}; the table stands"

    new_state = state.copy()
    stored_ts = np.asarray(table_pq["box.timestamps"].combine_chunks().flatten().to_numpy(), dtype=np.float64)
    stored_ts = stored_ts.reshape(-1, ts_width)
    new_ts = stored_ts.copy()
    report: dict[str, Any] = {"episodes": {}}
    for ep_index in sorted({int(x) for x in episode_col}):
        ep_dir = dataset / "episodes" / f"episode_{ep_index:06d}"
        rows_idx = np.nonzero(episode_col == ep_index)[0]
        ep = load_episode(ep_dir)
        box_ids = lr3.box_ids_from_inputs([], ep["samples"])
        if len(lr3.box_state_names(box_ids)) != width:
            raise ValueError(
                f"{ep_dir.name}: {len(box_ids)} BOX(es) in the raw data, the parquet has {width} state channels"
            )
        prior = ((info.get("box_alignment") or {}).get("episodes") or {}).get(str(ep_index), {}).get("boxes")
        legacy = build_rows(ep, len(rows_idx), fps, box_ids, legacy=True, prior=prior)
        same, worst = _reproduces(
            legacy, state[rows_idx], stored_ts[rows_idx], lr3.box_timestamp_names(box_ids)
        )
        if same is None:
            raise RuntimeError(
                f"{ep_dir.name}: rebuilding the recorded pairing does not reproduce the stored columns "
                f"({worst}); not realigning a dataset this tool cannot reproduce"
            )
        choice: dict[str, dict[str, Any]] = {}
        delays_s: dict[str, float] = {}
        for box in box_ids:
            own = measured.get(ep_index, {}).get(box)
            if own is not None and own["r"] >= MIN_PEAK_CORRELATION:
                delays_s[box] = own["d_box_ms"] / 1000.0
                choice[box or "unnamed"] = {"target": "mid_exposure_plus_d_box", "source": "episode", **own}
            elif box and box in table:
                delays_s[box] = table[box]
                choice[box] = {
                    "target": "mid_exposure_plus_d_box",
                    "source": "table",
                    "d_box_ms": round(table[box] * 1000.0, 3),
                    **({"episode_estimate": own} if own else {}),
                }
            else:
                choice[box or "unnamed"] = {
                    "target": "raw_sof",
                    "source": "unmeasured",
                    **({"episode_estimate": own} if own else {}),
                }
        rows = build_rows(ep, len(rows_idx), fps, box_ids, delays_s=delays_s)
        new_state[rows_idx] = np.asarray([r["observation.state"] for r in rows], dtype=np.float64)
        new_ts[rows_idx] = np.asarray([r["box.timestamps"] for r in rows], dtype=np.float64)
        moved = np.abs(np.asarray(new_state[rows_idx], dtype=np.float32) - state[rows_idx])
        report["episodes"][str(ep_index)] = {
            "boxes": choice,
            "reproduced_same_pick": same,
            "max_change": float(moved.max()),
        }

    report.update(
        {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "exposure_fraction": -0.5,
            "interpolate": True,
            "min_peak_correlation": MIN_PEAK_CORRELATION,
            "delay_table": lr3.BOX_TIME_DELAYS_PATH.name,
            "estimate": estimate_note,
            "tool": "tools/thor/gmsl2/box_realign.py",
        }
    )
    if not write:
        return report

    for path in (data_path, dataset / "meta" / "stats.json"):
        backup = path.with_name(path.name + ".pre_realign")
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
    state_type = table_pq.schema.field("observation.state").type
    ts_type = table_pq.schema.field("box.timestamps").type
    out = table_pq.set_column(
        table_pq.schema.get_field_index("observation.state"),
        "observation.state",
        pa.array(new_state.astype(np.float32).tolist(), type=state_type),
    )
    out = out.set_column(
        out.schema.get_field_index("box.timestamps"),
        "box.timestamps",
        pa.array(new_ts.tolist(), type=ts_type),
    )
    pq.write_table(out, data_path)
    stats_path = dataset / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats["observation.state"] = lr3._table_column_stats(out, "observation.state", width=width)
    stats["box.timestamps"] = lr3._table_column_stats(out, "box.timestamps", width=ts_width)
    stats_path.write_text(json.dumps(stats, indent=4), encoding="utf-8")
    info["box_alignment"] = report
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--estimate", action="store_true", help="measure each episode's d_box from its tracking"
    )
    parser.add_argument(
        "--write", action="store_true", help="rewrite the BOX columns (otherwise only report)"
    )
    args = parser.parse_args(argv)
    failed = 0
    for dataset in args.datasets:
        try:
            report = realign(dataset, estimate=args.estimate, write=args.write)
        except Exception as exc:  # noqa: BLE001 - one dataset's problem is reported, the rest go on
            print(f"{dataset.name}: NOT realigned: {exc}")
            failed += 1
            continue
        for ep, entry in report["episodes"].items():
            boxes = "; ".join(
                f"{box} {c['target']}" + (f" {c['d_box_ms']} ms ({c['source']})" if "d_box_ms" in c else "")
                for box, c in entry["boxes"].items()
            )
            print(f"{dataset.name} ep {ep}: {boxes}; state moved up to {entry['max_change']:.4g}")
        print(
            f"{dataset.name}: {'written' if args.write else 'dry run, nothing written'} "
            f"(per-episode estimate: {report['estimate']})"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
