"""The frame-time semantics resolver must read the convention off the data.

The two hypotheses differ by a whole exposure, so the resolver is only useful
if it distinguishes them on data whose answer we already know.  These build
sidecars under each hypothesis and check it says the right thing -- and, just
as importantly, that it refuses when the data cannot answer.
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.thor.gmsl2 import resolve_frame_time_semantics as rfts  # noqa: E402
from tools.thor.gmsl2.argus_frame_sync import (  # noqa: E402
    ArgusFrameMetadata,
    frame_metadata_sidecar_path,
    write_frame_metadata_csv,
)

FPS = 60
PERIOD_NS = 1_000_000_000 // FPS
TSC_OFFSET_NS = 26_600_000_000  # the measured monotonic -> TSC domain offset
READOUT_NS = 9_000_000          # a plausible fixed readout time


def _exposures_ns(n, rng):
    """An auto-exposure walk: 4 ms to 12 ms, the range the roadmap prices."""
    return [int(rng.uniform(4_000_000, 12_000_000)) for _ in range(n)]


def _write(ep_dir, camera, *, sof_is_integration_start, n=300, seed=0):
    rng = random.Random(seed)
    exposures = _exposures_ns(n, rng)
    rows = []
    for i, exp in enumerate(exposures):
        integration_start = 1_000_000_000 + i * PERIOD_NS
        if sof_is_integration_start:
            # SOF..EOF brackets integration.
            sof = integration_start
            eof = integration_start + exp
        else:
            # SOF..EOF brackets readout, which begins when integration ends.
            sof = integration_start + exp
            eof = sof + READOUT_NS
        rows.append(
            ArgusFrameMetadata(
                camera=camera,
                encoded_frame_index=i,
                local_frame_number=i,
                sensor_timestamp_ns=sof,               # same event, other domain
                sof_tsc_ns=sof + TSC_OFFSET_NS,
                eof_tsc_ns=eof + TSC_OFFSET_NS,
                internal_frame_count=i,
                sensor_exposure_time_ns=exp,
                sensor_analog_gain=1.0,
            )
        )
    write_frame_metadata_csv(frame_metadata_sidecar_path(ep_dir, camera), rows)


def test_resolves_sof_as_start_of_integration(tmp_path):
    _write(tmp_path, "cam_00", sof_is_integration_start=True)
    assert rfts.main([str(tmp_path)]) == 0
    v = rfts.analyse(frame_metadata_sidecar_path(tmp_path, "cam_00"))
    assert v.sof_eof.slope == pytest.approx(1.0, abs=1e-6)
    assert rfts._verdict_text(v)[1] == pytest.approx(0.5)


def test_resolves_sof_as_start_of_readout(tmp_path):
    """The MIPI reading: SOF is transmission, so it follows integration."""
    _write(tmp_path, "cam_00", sof_is_integration_start=False)
    assert rfts.main([str(tmp_path)]) == 0
    v = rfts.analyse(frame_metadata_sidecar_path(tmp_path, "cam_00"))
    # EOF - SOF is the fixed readout time, carrying no exposure dependence.
    assert v.sof_eof.slope == pytest.approx(0.0, abs=1e-6)
    assert v.sof_eof.median_us == pytest.approx(READOUT_NS / 1000.0, abs=1.0)
    assert rfts._verdict_text(v)[1] == pytest.approx(-0.5)


def test_locked_exposure_cannot_reveal_its_own_convention(tmp_path):
    """A pinned exposure is the one recording that cannot answer this.

    Worth a test because it is counter-intuitive: locking is good for the
    measurement and useless for the calibration of the measurement.
    """
    rows = [
        ArgusFrameMetadata(
            camera="cam_00", encoded_frame_index=i, local_frame_number=i,
            sensor_timestamp_ns=1_000_000_000 + i * PERIOD_NS,
            sof_tsc_ns=1_000_000_000 + i * PERIOD_NS + TSC_OFFSET_NS,
            eof_tsc_ns=1_000_000_000 + i * PERIOD_NS + TSC_OFFSET_NS + READOUT_NS,
            internal_frame_count=i, sensor_exposure_time_ns=8_000_000,
            sensor_analog_gain=1.0,
        )
        for i in range(300)
    ]
    write_frame_metadata_csv(frame_metadata_sidecar_path(tmp_path, "cam_00"), rows)
    assert rfts.main([str(tmp_path)]) == 2


def test_missing_exposure_column_is_unresolvable_not_zero(tmp_path):
    rows = [
        ArgusFrameMetadata(
            camera="cam_00", encoded_frame_index=i, local_frame_number=i,
            sensor_timestamp_ns=1_000_000_000 + i * PERIOD_NS,
            sof_tsc_ns=1_000_000_000 + i * PERIOD_NS + TSC_OFFSET_NS,
            eof_tsc_ns=1_000_000_000 + i * PERIOD_NS + TSC_OFFSET_NS + READOUT_NS,
            internal_frame_count=i,
        )
        for i in range(300)
    ]
    write_frame_metadata_csv(frame_metadata_sidecar_path(tmp_path, "cam_00"), rows)
    assert rfts.main([str(tmp_path)]) == 2


def test_cameras_disagreeing_is_a_data_problem_not_a_setting(tmp_path):
    """One SoC cannot hold two conventions, so disagreement must fail loudly."""
    _write(tmp_path, "cam_00", sof_is_integration_start=True, seed=1)
    _write(tmp_path, "cam_01", sof_is_integration_start=False, seed=2)
    assert rfts.main([str(tmp_path)]) == 1


def test_sensor_timestamp_marking_a_different_event_flips_the_sign(tmp_path):
    """If sensor_timestamp_ns and SOF are an exposure apart, they are not the
    same event, and the fraction that applies to SOF is wrong for the column we
    actually use."""
    rng = random.Random(7)
    rows = []
    for i in range(300):
        exp = int(rng.uniform(4_000_000, 12_000_000))
        integration_start = 1_000_000_000 + i * PERIOD_NS
        sof = integration_start          # SOF = start of integration
        rows.append(
            ArgusFrameMetadata(
                camera="cam_00", encoded_frame_index=i, local_frame_number=i,
                # ...but sensor_timestamp_ns is latched at the end of it.
                sensor_timestamp_ns=integration_start + exp,
                sof_tsc_ns=sof + TSC_OFFSET_NS,
                eof_tsc_ns=sof + exp + TSC_OFFSET_NS,
                internal_frame_count=i, sensor_exposure_time_ns=exp,
                sensor_analog_gain=1.0,
            )
        )
    write_frame_metadata_csv(frame_metadata_sidecar_path(tmp_path, "cam_00"), rows)
    v = rfts.analyse(frame_metadata_sidecar_path(tmp_path, "cam_00"))
    assert v.sof_eof.slope == pytest.approx(1.0, abs=1e-6)     # SOF..EOF = integration
    assert v.sens_sof.slope == pytest.approx(-1.0, abs=1e-6)   # but they differ by exp
    text, frac = rfts._verdict_text(v)
    assert "DIFFERENT events" in text
    assert frac == pytest.approx(-0.5)
