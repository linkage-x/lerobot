#!/usr/bin/env python3
"""Work out, from recorded data, which instant the Argus frame stamps refer to.

The correction from a frame stamp to the instant the scene was actually
sampled is ``fraction * exposure``, and ``fraction`` is +0.5 or -0.5 depending
on a question nobody has answered from measurement:

    does ``getSensorSofTimestampTsc`` mark the start of *integration*,
    or the start of *readout* (i.e. the end of integration)?

"SOF" is a MIPI CSI-2 transport event -- the sensor beginning to *transmit*
line 1 -- which on a global-shutter part happens after integration finishes.
But Argus may equally be reporting a sensor-reported exposure-start from the
frame descriptor.  The two readings differ by a whole exposure, so guessing is
not an option: at 680 mm/s a 8.7 ms exposure is 5.9 mm, and a sign error
doubles it.

Nothing new has to be recorded to settle it.  Every sidecar row already carries
three stamps, and the intervals between them either track the exposure or they
do not::

    d_sof_eof = eof_tsc_ns - sof_tsc_ns
    d_sens_sof = sof_tsc_ns - sensor_timestamp_ns     (fixed domain offset removed)

Regress each against ``sensor_exposure_time_ns``:

    slope ~ +1  ->  that interval *is* the integration window
    slope ~  0  ->  that interval is a fixed hardware time (readout), and the
                    two stamps bracket readout rather than exposure

The leverage comes from the exposure *moving*, so this wants an ordinary
auto-exposure recording -- a locked one carries no information about its own
semantics.  Point it at a varied-lighting episode and it answers in one pass.

    python -m tools.thor.gmsl2.resolve_frame_time_semantics <episode_dir>

Exit codes: 0 resolved, 1 the data is inconsistent with either model, 2 it
could not be resolved (no exposure column, or the exposure never moved).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .argus_frame_sync import SIDECAR_BASENAME, read_frame_metadata_csv
except ImportError:  # running as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from argus_frame_sync import SIDECAR_BASENAME, read_frame_metadata_csv

MIN_EXPOSURE_SPREAD_US = 500.0
"""Below this the regression has no leverage and the slope is noise.

Half a millisecond of exposure swing moves mid-exposure by 250 us, which is
already comparable to the 293 us random timing term -- so if the exposure moved
less than this, the answer would be an artefact of whatever else drifted."""

SLOPE_TOLERANCE = 0.25
"""How far from the ideal 0 or +1 a slope may sit and still be called.

The two hypotheses are a full unit apart, so this is a wide gate that still
cannot confuse them; anything landing between the two is reported as neither
rather than rounded to the nearer one."""


@dataclass(frozen=True)
class Regression:
    slope: float
    r2: float
    n: int
    median_us: float
    spread_us: float


def _regress(x: list[float], y: list[float]) -> Regression:
    n = len(x)
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx if sxx > 0 else float("nan")
    syy = sum((v - my) ** 2 for v in y)
    r2 = (sxy * sxy) / (sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    ordered = sorted(y)
    lo = ordered[int(0.05 * (n - 1))]
    hi = ordered[int(0.95 * (n - 1))]
    return Regression(slope, r2, n, statistics.median(y), hi - lo)


def _classify(slope: float) -> str:
    """Classify on |slope|, because the sign only records which way the
    difference was taken.  ``sof - sensor_ts`` is ``-exposure`` when SOF marks
    the start and ``sensor_ts`` the end, and ``+exposure`` the other way round;
    both mean the same thing -- the two stamps are a whole exposure apart and
    therefore mark different events."""
    if abs(abs(slope) - 1.0) <= SLOPE_TOLERANCE:
        return "tracks_exposure"
    if abs(slope) <= SLOPE_TOLERANCE:
        return "fixed"
    return "neither"


@dataclass(frozen=True)
class CameraVerdict:
    camera: str
    exposure_spread_us: float
    sof_eof: Regression | None
    sens_sof: Regression | None
    note: str = ""


def analyse(path: Path) -> CameraVerdict:
    rows = read_frame_metadata_csv(path)
    camera = rows[0].camera if rows else path.name.split(".")[0]
    usable = [
        r for r in rows
        if r.sensor_exposure_time_ns > 0 and r.sof_tsc_ns > 0 and r.sensor_timestamp_ns > 0
    ]
    if len(usable) < 30:
        return CameraVerdict(camera, 0.0, None, None,
                             f"only {len(usable)} rows carry exposure + both stamps")

    exposure_us = [r.sensor_exposure_time_ns / 1000.0 for r in usable]
    ordered = sorted(exposure_us)
    spread = ordered[int(0.95 * (len(ordered) - 1))] - ordered[int(0.05 * (len(ordered) - 1))]
    if spread < MIN_EXPOSURE_SPREAD_US:
        return CameraVerdict(
            camera, spread, None, None,
            f"exposure only moved {spread:.0f} us (need {MIN_EXPOSURE_SPREAD_US:.0f}); "
            "a locked recording cannot reveal its own convention -- use an "
            "auto-exposure episode with varied lighting",
        )

    # EOF - SOF. Both are TSC, so this needs no domain correction.
    have_eof = all(r.eof_tsc_ns > 0 for r in usable)
    sof_eof = None
    if have_eof:
        sof_eof = _regress(
            exposure_us, [(r.eof_tsc_ns - r.sof_tsc_ns) / 1000.0 for r in usable]
        )

    # SOF(TSC) - sensor_timestamp(MONOTONIC). The domain offset is a constant,
    # so it cannot affect the slope; it is left in and shows up in the median.
    sens_sof = _regress(
        exposure_us, [(r.sof_tsc_ns - r.sensor_timestamp_ns) / 1000.0 for r in usable]
    )
    return CameraVerdict(camera, spread, sof_eof, sens_sof)


def _verdict_text(v: CameraVerdict) -> tuple[str, float | None]:
    """Return (human verdict, implied exposure fraction for sensor_timestamp_ns)."""
    if v.sof_eof is None:
        return ("cannot resolve: " + (v.note or "no EOF column"), None)
    sof_eof = _classify(v.sof_eof.slope)
    sens_sof = _classify(v.sens_sof.slope) if v.sens_sof else "unknown"

    if sof_eof == "tracks_exposure":
        base = ("SOF..EOF is the INTEGRATION window, so SOF is the start of "
                "integration and mid-exposure is half an exposure LATER")
        frac = 0.5
    elif sof_eof == "fixed":
        base = (f"SOF..EOF is a FIXED {v.sof_eof.median_us:.0f} us window, i.e. readout. "
                "SOF is therefore the start of readout = the END of integration, "
                "and mid-exposure is half an exposure EARLIER")
        frac = -0.5
    else:
        return (f"inconsistent: EOF-SOF moves {v.sof_eof.slope:.2f} us per us of "
                "exposure, which is neither model", None)

    if sens_sof == "fixed":
        base += (". sensor_timestamp_ns tracks SOF at a fixed offset, so the same "
                 "fraction applies to it")
    elif sens_sof == "tracks_exposure":
        base += (". WARNING: sensor_timestamp_ns and SOF are a whole exposure apart, "
                 "so they mark DIFFERENT events -- the fraction above describes SOF, "
                 "and sensor_timestamp_ns needs the opposite sign")
        frac = -frac
    return (base, frac)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("episode_dir", type=Path)
    args = p.parse_args(argv)

    sidecars = sorted(args.episode_dir.glob(f"*.{SIDECAR_BASENAME}"))
    if not sidecars:
        print(f"cannot resolve: no *.{SIDECAR_BASENAME} under {args.episode_dir}",
              file=sys.stderr)
        return 2

    verdicts = [analyse(pth) for pth in sidecars]
    fractions: list[float] = []
    unresolved: list[str] = []
    inconsistent: list[str] = []

    for v in verdicts:
        print(f"\n=== {v.camera} ===")
        if v.sof_eof is None and v.sens_sof is None:
            print(f"  {v.note}")
            unresolved.append(v.camera)
            continue
        print(f"  exposure p05..p95 spread : {v.exposure_spread_us:8.0f} us")
        if v.sof_eof:
            print(f"  (EOF - SOF)   vs exposure: slope {v.sof_eof.slope:+.3f}  "
                  f"r2 {v.sof_eof.r2:.3f}  median {v.sof_eof.median_us:.0f} us")
        if v.sens_sof:
            print(f"  (SOF - sensor_ts) vs exp : slope {v.sens_sof.slope:+.3f}  "
                  f"r2 {v.sens_sof.r2:.3f}  spread {v.sens_sof.spread_us:.0f} us")
        text, frac = _verdict_text(v)
        print(f"  -> {text}")
        if frac is None:
            (inconsistent if v.sof_eof else unresolved).append(v.camera)
        else:
            fractions.append(frac)
            print(f"  -> EXPOSURE_CENTER_FRACTION = {frac:+.1f}")

    print()
    if inconsistent:
        print(f"FAIL: {', '.join(inconsistent)} match neither model. Do not apply a "
              "correction until this is understood -- a wrong sign costs a whole "
              "exposure, which is worse than no correction at all.", file=sys.stderr)
        return 1
    if not fractions:
        print("cannot resolve: " + "; ".join(unresolved), file=sys.stderr)
        return 2
    if len(set(fractions)) > 1:
        print("FAIL: cameras disagree about the convention, which should be impossible "
              "on one SoC. Treat this as a data problem, not a per-camera setting.",
              file=sys.stderr)
        return 1

    frac = fractions[0]
    print(f"RESOLVED on {len(fractions)} camera(s): "
          f"thor_lerobot_v3.EXPOSURE_CENTER_FRACTION should be {frac:+.1f}")
    if unresolved:
        print(f"(no verdict from: {', '.join(unresolved)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
