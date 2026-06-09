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
