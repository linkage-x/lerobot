"""Synthetic end-to-end alignment tests for the Thor BOX↔camera soft-sync.

Bucket B.1 from the soft-sync feasibility review: these exercise the full
L3a/L3b alignment math in ``thor_lerobot_v3`` with purely *synthetic* BOX
samples, so they need **no handheld BOX hardware**. They pin the alignment
correctness as a regression test before real-machine debugging is possible.

The synthetic model (see tools/thor/ts_sync.md §4–§5):

  - one sensor on a *known* MCU clock: ``host_wall = SLOPE_TRUE * mcu_ts + T0``
    (mcu_ts in milliseconds, sensor at 200 Hz → 5 ms between samples),
  - deterministic poll jitter injected into the recorded wall times,
  - a known camera pipeline-start delay (``pts_offset``).

The gripper's ``distance_m`` maps to ``observation.state[0]`` (see
``box_snapshot_to_state``), so each synthetic sample carries
``distance_m = its sample index``. Every assertion can therefore read back
*exactly which sample* a given camera frame selected.
"""

import pytest

from tools.thor.gmsl2 import thor_lerobot_v3 as lr3

FPS = 60
T0_WALL_S = 1000.0
SLOPE_TRUE = 0.001          # seconds per MCU tick (mcu_ts is in milliseconds)
SAMPLE_PERIOD_TICKS = 5     # 200 Hz: 5 ms between samples
N_SAMPLES = 120             # covers 0.595 s of true time (>= 10 → calibration runs)
DURATION_S = 0.5
FRAME_COUNT = round(DURATION_S * FPS)   # 30 camera frames
GRIPPER = "box_gripper"


def _true_rel_s(k: int) -> float:
    """Ground-truth host time (relative to t0) of sample ``k``, no jitter."""
    return SLOPE_TRUE * (k * SAMPLE_PERIOD_TICKS)


def _alt_jitter(k: int) -> float:
    """Deterministic ±4 ms poll jitter that flips the apparent order of
    adjacent 5 ms-spaced samples, corrupting raw nearest-neighbor selection."""
    return 0.004 if k % 2 == 0 else -0.004


def _make_gripper_samples(n: int, jitter=None) -> dict:
    """Build ``n`` samples at 200 Hz on a clean MCU clock.

    Shape matches what ``thor_record`` hands to ``Lr3Writer.append_episode``:
    ``{sid: [{"t_rel_s", "wall_s", "data": {"timestamp", "distance_m"}}]}``.
    """
    samples = []
    for k in range(n):
        mcu_ts = k * SAMPLE_PERIOD_TICKS
        true_wall = T0_WALL_S + _true_rel_s(k)
        wall_s = true_wall + (0.0 if jitter is None else jitter(k))
        samples.append({
            "t_rel_s": wall_s - T0_WALL_S,
            "wall_s": wall_s,
            "data": {"timestamp": mcu_ts, "distance_m": float(k)},
        })
    return {GRIPPER: samples}


def _ground_truth(pts_offset: float) -> list[int]:
    """Per-frame selected sample index using the *true* (jitter-free) clock.

    Computed via the same ``_nearest_sample_data`` primitive the writer uses,
    so tie-breaking matches exactly.
    """
    times = [_true_rel_s(k) for k in range(N_SAMPLES)]
    smp = [{"data": {"distance_m": float(k)}} for k in range(N_SAMPLES)]
    out = []
    for f in range(FRAME_COUNT):
        t = pts_offset + f / FPS
        out.append(int(round(lr3._nearest_sample_data(times, smp, t)["distance_m"])))
    return out


def _raw_distances(samples: dict, pts_offset: float) -> list[int]:
    """Per-frame selection using the recorded (jittered, *uncalibrated*) times —
    i.e. what the alignment would pick if MCU calibration were disabled."""
    js = sorted(samples[GRIPPER], key=lambda s: s["t_rel_s"])
    jtimes = [s["t_rel_s"] for s in js]
    out = []
    for f in range(FRAME_COUNT):
        t = pts_offset + f / FPS
        out.append(int(round(lr3._nearest_sample_data(jtimes, js, t)["distance_m"])))
    return out


def _aligned_distances(samples: dict, pts_offset: float, t0: float = T0_WALL_S) -> list[int]:
    """Per-frame ``observation.state[0]`` produced by the real writer path
    (``_build_episode_rows`` → calibrate → nearest-neighbor)."""
    rows = lr3._build_episode_rows(
        fps=FPS,
        episode_index=0,
        snapshots=[],
        duration_s=DURATION_S,
        sensor_samples=samples,
        t0_wall_s=t0,
        pts_offset_s=pts_offset,
    )
    return [int(round(r["observation.state"][0])) for r in rows]


# --------------------------------------------------------------------------
# 1. nearest-neighbor primitive
# --------------------------------------------------------------------------

def test_nearest_sample_data_primitive():
    times = [0.0, 0.10, 0.20, 0.30]
    smp = [{"data": {"distance_m": float(i)}} for i in range(4)]

    # before the first sample → clamp to first
    assert lr3._nearest_sample_data(times, smp, -1.0)["distance_m"] == 0.0
    # after the last sample → clamp to last
    assert lr3._nearest_sample_data(times, smp, 9.0)["distance_m"] == 3.0
    # clearly closer to index 2
    assert lr3._nearest_sample_data(times, smp, 0.19)["distance_m"] == 2.0
    # exact tie (0.15 between 0.10 and 0.20) → bisect prefers the earlier sample
    assert lr3._nearest_sample_data(times, smp, 0.15)["distance_m"] == 1.0
    # empty input is safe
    assert lr3._nearest_sample_data([], [], 0.0) == {}


# --------------------------------------------------------------------------
# 2. pts_offset shifts the frame-time grid (camera pipeline-start correction)
# --------------------------------------------------------------------------

def test_pts_offset_shifts_frame_grid():
    clean = _make_gripper_samples(N_SAMPLES)

    # No camera-start delay: frame grid is exactly N / fps.
    assert _aligned_distances(clean, 0.0) == _ground_truth(0.0)
    # A 12 ms pipeline-start delay shifts every frame's sampling point.
    assert _aligned_distances(clean, 0.012) == _ground_truth(0.012)
    # The offset must actually change which samples get selected.
    assert _aligned_distances(clean, 0.0) != _aligned_distances(clean, 0.012)

    # The emitted per-frame timestamp grid is pts_offset + frame / fps.
    rows = lr3._build_episode_rows(
        fps=FPS, episode_index=0, snapshots=[], duration_s=DURATION_S,
        sensor_samples=clean, t0_wall_s=T0_WALL_S, pts_offset_s=0.012,
    )
    assert rows[0]["timestamp"] == pytest.approx(0.012)
    assert rows[5]["timestamp"] == pytest.approx(0.012 + 5 / FPS)
    assert len(rows) == FRAME_COUNT


# --------------------------------------------------------------------------
# 3. MCU clock calibration removes poll jitter (the heart of L3b)
# --------------------------------------------------------------------------

def test_mcu_calibration_removes_poll_jitter():
    jit = _make_gripper_samples(N_SAMPLES, jitter=_alt_jitter)
    truth = _ground_truth(0.0)

    # Raw (uncalibrated) selection is corrupted by the injected jitter ...
    assert _raw_distances(jit, 0.0) != truth
    # ... but after MCU↔host calibration every frame recovers the true sample.
    assert _aligned_distances(jit, 0.0) == truth

    # The linear fit recovers the underlying MCU clock with sub-ms residual,
    # so calibration is accepted (does not fall back to poll times).
    mcu = [s["data"]["timestamp"] for s in jit[GRIPPER]]
    wall = [s["wall_s"] for s in jit[GRIPPER]]
    slope, intercept, res_std = lr3.calibrate_mcu_clock(mcu, wall)
    assert slope == pytest.approx(SLOPE_TRUE, abs=1e-5)
    assert intercept == pytest.approx(T0_WALL_S, abs=1e-3)
    assert res_std < 0.01            # well under the 0.05 s fallback threshold


# --------------------------------------------------------------------------
# 4. calibration safety fallbacks keep the original poll-based times
# --------------------------------------------------------------------------

def test_calibration_fallback_too_few_samples():
    few = _make_gripper_samples(5)            # < 10 → cannot fit
    out = lr3.calibrate_sensor_samples(few, T0_WALL_S)
    assert [s["t_rel_s"] for s in out[GRIPPER]] == [s["t_rel_s"] for s in few[GRIPPER]]


def test_calibration_fallback_all_zero_mcu_timestamps():
    zero = _make_gripper_samples(40)
    for s in zero[GRIPPER]:
        s["data"]["timestamp"] = 0          # sensor never reported MCU time
    out = lr3.calibrate_sensor_samples(zero, T0_WALL_S)
    assert [s["t_rel_s"] for s in out[GRIPPER]] == [s["t_rel_s"] for s in zero[GRIPPER]]


def test_calibration_fallback_poor_fit():
    bad = _make_gripper_samples(40)
    # Wall times have no linear relationship to MCU ticks (±200 ms swing) →
    # residual_std >> 0.05 s → keep poll-based times.
    for i, s in enumerate(bad[GRIPPER]):
        s["wall_s"] = T0_WALL_S + (0.2 if i % 2 else -0.2)
        s["t_rel_s"] = s["wall_s"] - T0_WALL_S
    out = lr3.calibrate_sensor_samples(bad, T0_WALL_S)
    assert [s["t_rel_s"] for s in out[GRIPPER]] == [s["t_rel_s"] for s in bad[GRIPPER]]


# --------------------------------------------------------------------------
# 5. full end-to-end: jittered samples → calibrate → align → parquet
# --------------------------------------------------------------------------

def test_end_to_end_alignment_via_parquet(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pts_offset = 0.010
    jit = _make_gripper_samples(N_SAMPLES, jitter=_alt_jitter)

    writer = lr3.Lr3Writer(tmp_path, repo_id="thor_sync_test", task="pick", fps=FPS)
    writer.append_episode(
        episode_index=0,
        snapshots=[],
        duration_s=DURATION_S,
        sensor_samples=jit,
        t0_wall_s=T0_WALL_S,
        pts_offset_s=pts_offset,
    )
    writer.finalize()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == FRAME_COUNT

    states = table["observation.state"].to_pylist()
    distances = [int(round(s[0])) for s in states]
    assert distances == _ground_truth(pts_offset)

    timestamps = table["timestamp"].to_pylist()
    assert timestamps[0] == pytest.approx(pts_offset, abs=1e-4)
    assert timestamps[5] == pytest.approx(pts_offset + 5 / FPS, abs=1e-4)


# --------------------------------------------------------------------------
# 6. schema contract: MCU timestamps live in metadata, not observation.state
# --------------------------------------------------------------------------

def test_timestamps_excluded_from_state_and_emitted_as_metadata_column(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")

    # observation.state must carry no timestamp/wall-clock channels and only
    # one gripper distance channel (the former duplicate ``gripper.pos`` is gone).
    assert "gripper.pos" not in lr3.BOX_STATE_NAMES
    assert lr3.BOX_STATE_NAMES[0] == "box_gripper.distance_m"
    assert sum(n == "box_gripper.distance_m" for n in lr3.BOX_STATE_NAMES) == 1
    assert not any(
        n.endswith("timestamp") or n.endswith("received_wall_time_s")
        for n in lr3.BOX_STATE_NAMES
    )
    # ...and they are preserved in the dedicated metadata vector instead.
    assert all(
        n.endswith("timestamp") or n.endswith("received_wall_time_s")
        for n in lr3.BOX_TIMESTAMP_NAMES
    )
    assert lr3.BOX_TIMESTAMP_NAMES[0] == "box_gripper.timestamp"

    clean = _make_gripper_samples(N_SAMPLES)
    writer = lr3.Lr3Writer(tmp_path, repo_id="thor_sync_test", task="pick", fps=FPS)
    writer.append_episode(
        episode_index=0, snapshots=[], duration_s=DURATION_S,
        sensor_samples=clean, t0_wall_s=T0_WALL_S, pts_offset_s=0.0,
    )
    writer.finalize()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert len(table["observation.state"].to_pylist()[0]) == len(lr3.BOX_STATE_NAMES)

    # The MCU timestamp rides in box.timestamps (gripper at column 0), not state,
    # and is stored float64 so the µs counter keeps full integer precision.
    box_ts = table["box.timestamps"].to_pylist()
    assert len(box_ts[0]) == len(lr3.BOX_TIMESTAMP_NAMES)
    gripper_mcu = [int(round(row[0])) for row in box_ts]
    # By construction distance_m == sample index k and data.timestamp == k*ticks,
    # so the aligned gripper MCU stamp is the selected index times the period.
    aligned_idx = _aligned_distances(clean, 0.0)
    expected_mcu = [idx * SAMPLE_PERIOD_TICKS for idx in aligned_idx]
    assert gripper_mcu == expected_mcu



def test_multi_box_namespaced_snapshots_expand_state_and_timestamps(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")

    snapshots = [
        {
            "t_relative_s": 0.0,
            "sensors": {
                "box1672693301/box_gripper": {"timestamp": 100.0, "distance_m": 0.031},
                "box1672693301/box_trigger": {"timestamp": 101.0, "travel_pct": 0.25},
                "box1819152274/box_gripper": {"timestamp": 200.0, "distance_m": 0.044},
                "box1819152274/box_trigger": {"timestamp": 201.0, "travel_pct": 0.75},
            },
        }
    ]
    box_ids = lr3.box_ids_from_snapshots(snapshots)
    assert box_ids == ("box1672693301", "box1819152274")

    rows = lr3._build_episode_rows(
        fps=FPS,
        episode_index=0,
        snapshots=snapshots,
        duration_s=1 / FPS,
        box_ids=box_ids,
    )
    state_names = lr3.box_state_names(box_ids)
    ts_names = lr3.box_timestamp_names(box_ids)
    assert len(rows[0]["observation.state"]) == len(lr3.BOX_STATE_NAMES) * 2
    assert len(rows[0]["box.timestamps"]) == len(lr3.BOX_TIMESTAMP_NAMES) * 2
    assert state_names[0] == "box1672693301.box_gripper.distance_m"
    assert "box1819152274.box_trigger.travel_pct" in state_names
    first_gripper = state_names.index("box1672693301.box_gripper.distance_m")
    second_gripper = state_names.index("box1819152274.box_gripper.distance_m")
    assert rows[0]["observation.state"][first_gripper] == pytest.approx(0.031)
    assert rows[0]["observation.state"][second_gripper] == pytest.approx(0.044)

    writer = lr3.Lr3Writer(tmp_path, repo_id="thor_multi_box_test", task="pick", fps=FPS)
    writer.append_episode(episode_index=0, snapshots=snapshots, duration_s=1 / FPS)
    writer.finalize()
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    assert len(table["observation.state"].to_pylist()[0]) == len(state_names)
    info = __import__("json").loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["names"] == list(state_names)
    assert info["features"]["box.timestamps"]["names"] == list(ts_names)


# --------------------------------------------------------------------------
# 7. frame_times_s: BOX nearest-neighbor targets the camera hardware SOF time
#    (online-sync sidecar) while the timestamp column stays N/fps. Sidecar
#    gaps/tail extrapolate the SOF fit (single basis, no N/fps splice).
#    Regression for the per-episode camera skew found on real Thor data
#    (-11..-53 ms; ts_sync.md §5.4 / experiments/ts_sync_skew_20260716/).
# --------------------------------------------------------------------------

DELTA_SKEW_S = -0.030   # a realistic per-episode camera skew (frame 0 before t0)


def _distances_with_frame_times(frame_times):
    rows = lr3._build_episode_rows(
        fps=FPS, episode_index=0, snapshots=[], duration_s=DURATION_S,
        sensor_samples=_make_gripper_samples(N_SAMPLES), t0_wall_s=T0_WALL_S,
        pts_offset_s=None, frame_times_s=frame_times,
    )
    return rows


def test_frame_times_shift_box_lookup_but_not_timestamp():
    frame_times = [DELTA_SKEW_S + f / FPS for f in range(FRAME_COUNT)]
    rows = _distances_with_frame_times(frame_times)

    # The BOX sample chosen for each frame is the one nearest the TRUE capture
    # time (grid shifted by the skew), not the idealized N/fps instant.
    assert [int(round(r["observation.state"][0])) for r in rows] == _ground_truth(DELTA_SKEW_S)
    # ...and that genuinely differs from the uncorrected N/fps selection.
    assert _ground_truth(DELTA_SKEW_S) != _ground_truth(0.0)
    # The emitted timestamp grid is UNCHANGED (loader contract: frame N == N/fps).
    assert [r["timestamp"] for r in rows] == pytest.approx([f / FPS for f in range(FRAME_COUNT)])


def test_frame_times_none_matches_grid():
    base = _distances_with_frame_times(None)
    grid = [int(round(r["observation.state"][0])) for r in base]
    assert grid == _ground_truth(0.0)


def test_frame_times_no_usable_sof_uses_uniform_grid():
    # With <2 valid SOF entries there is no fit to extrapolate, so the WHOLE
    # episode stays on a single uniform N/fps grid (no two-basis splice).
    frame_times = [None] * FRAME_COUNT
    frame_times[3] = float("nan")
    rows = _distances_with_frame_times(frame_times)
    assert [int(round(r["observation.state"][0])) for r in rows] == _ground_truth(0.0)


def test_frame_times_interior_gaps_extrapolate_not_grid():
    # Interior gap frames are filled from the linear SOF fit (same basis),
    # NOT dropped to N/fps — so the result equals the fully-corrected grid.
    frame_times = [DELTA_SKEW_S + f / FPS for f in range(FRAME_COUNT)]
    for f in (4, 5, 17):
        frame_times[f] = None
    rows = _distances_with_frame_times(frame_times)
    assert [int(round(r["observation.state"][0])) for r in rows] == _ground_truth(DELTA_SKEW_S)


def test_frame_times_short_tail_extrapolates_single_basis():
    # A sidecar shorter than the camera clip must extrapolate the SOF fit for
    # the tail (single time basis), NOT splice N/fps back in. Because the
    # synthetic SOF is exactly linear, the extrapolated tail reproduces the
    # fully-corrected grid across ALL frames.
    frame_times = [DELTA_SKEW_S + f / FPS for f in range(FRAME_COUNT - 4)]
    rows = _distances_with_frame_times(frame_times)
    got = [int(round(r["observation.state"][0])) for r in rows]
    assert got == _ground_truth(DELTA_SKEW_S)
    # ...and this is genuinely different from the old N/fps-tail behavior.
    grid = _ground_truth(0.0)
    assert got[FRAME_COUNT - 4:] != grid[FRAME_COUNT - 4:]


# --------------------------------------------------------------------------
# 8. camera_frame_times_rel: read hardware SOF frame times from the online-sync
#    sidecar in the t0-relative (monotonic) domain shared with BOX t_rel_s.
# --------------------------------------------------------------------------

def _write_sidecar(ep_dir, cam, t0_mono, frame0_skew_s, n):
    ep_dir.mkdir(parents=True, exist_ok=True)
    header = ("camera,logical_frame_index,local_frame_number,sensor_timestamp_ns,"
              "sof_tsc_ns,eof_tsc_ns,internal_frame_count\n")
    lines = [header]
    for i in range(n):
        sens_ns = int((t0_mono + frame0_skew_s + i / FPS) * 1e9)
        sof_ns = sens_ns + 26_600_000_000            # separate TSC domain (must be ignored)
        lines.append(f"{cam},{i},{i + 33},{sens_ns},{sof_ns},{sof_ns + 14_000},{i + 62}\n")
    (ep_dir / f"{cam}.argus_frame_metadata.csv").write_text("".join(lines))


def test_camera_frame_times_rel_reads_sidecar(tmp_path):
    t0_mono = 6782.984511942
    # Two PWM-locked cameras share the same sensor_timestamp per logical frame.
    _write_sidecar(tmp_path, "cam_00", t0_mono, -0.0535, 4)
    _write_sidecar(tmp_path, "cam_06", t0_mono, -0.0535, 4)

    ft = lr3.camera_frame_times_rel(tmp_path, t0_mono)
    assert ft is not None and len(ft) == 4
    # time[N] = sensor_timestamp_ns/1e9 - t0_mono ; frame 0 is 53.5 ms before t0.
    assert ft[0] == pytest.approx(-0.0535, abs=1e-6)
    assert ft[1] == pytest.approx(-0.0535 + 1 / FPS, abs=1e-6)
    # sof_tsc_ns (separate 26.6 s-offset TSC domain) must NOT leak in.
    assert all(abs(t) < 1.0 for t in ft)


def test_camera_frame_times_rel_fallbacks(tmp_path):
    _write_sidecar(tmp_path, "cam_00", 6782.984511942, -0.02, 3)
    # Missing t0_mono → None (caller falls back to N/fps).
    assert lr3.camera_frame_times_rel(tmp_path, None) is None
    assert lr3.camera_frame_times_rel(tmp_path, 0.0) is None
    # No sidecar in the directory → None.
    assert lr3.camera_frame_times_rel(tmp_path / "no_such_ep", 6782.98) is None
