#!/usr/bin/env python3
"""Price what the exposure costs an episode's position labels, after correction.

This replaces the earlier ``check_exposure_lock.py``, and the change of name is
the point.  That script asked "did the exposure stay pinned?" and failed the
episode when it wandered.  Once the exposure is *recorded per frame*, wandering
is no longer a bias -- ``camera_times`` / ``thor_lerobot_v3`` subtract
``fraction * exposure`` frame by frame, so the label follows it.  Gating on
wander would now reject episodes that are already correct.

What is left after that correction is worth measuring, and it is not the same
list:

  1. **Was the exposure recorded at all.**  This is now a hard requirement
     rather than a nicety: the correction is only as good as the column, and an
     all-zero column is not a locked exposure, it is an unknown one.

  2. **Cross-camera exposure spread.**  Cameras holding different exposures
     sample different instants.  One fused pose carries one time, so this part
     does *not* come out in the correction -- it is a floor.  Equalising the
     exposures is the only thing that removes it.

  3. **Motion blur.**  ``exposure * |v|`` is how far the scene smears, which
     costs detection precision rather than timing.  It belongs here because the
     same column measures it, and because shortening the exposure is the one
     action that improves both 2 and 3.

  4. **The second-order term inside the window.**  The detected centroid is the
     time-average over the integration window, which equals mid-exposure exactly
     only for constant velocity; under acceleration ``a`` it is off by
     ``a * T^2 / 24``.  Reported so "mid-exposure" is a measured approximation
     rather than an article of faith.

    python -m tools.thor.gmsl2.check_exposure_timing <episode_dir>
    python -m tools.thor.gmsl2.check_exposure_timing <episode_dir> --speed-mm-s 1500

Exit codes: 0 within tolerance, 1 a finding about the recording, 2 it could not
be checked -- the same 0/1/2 split the laser-tracker CLIs use.
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

P95_HAND_SPEED_MM_S = 680.0
"""The rig's measured p95 hand speed.  Used as the default because a timing
error is only visible as ``|v| * dt``, so a millimetre figure needs a speed
attached or it means nothing."""

TYPICAL_HAND_ACCEL_M_S2 = 10.0
"""A brisk human hand reversal.  Only used to show the second-order term is
small; it enters as ``a * T^2 / 24`` and stays micrometric for any plausible
value, which is the useful conclusion."""


@dataclass(frozen=True)
class CameraExposure:
    camera: str
    n_frames: int
    n_reported: int
    median_us: float
    min_us: float
    max_us: float
    p95_abs_dev_us: float
    gain_min: float
    gain_max: float

    @property
    def span_us(self) -> float:
        return self.max_us - self.min_us

    @property
    def reported(self) -> bool:
        """Whether Argus gave us the field at all.

        Sidecars written before the column existed, or a driver that does not
        populate it, both show up as all-zero -- and an all-zero column would
        make the per-frame correction silently do nothing.
        """
        return self.n_reported > 0 and self.median_us > 0.0

    def blur_mm(self, speed_mm_s: float) -> float:
        return self.median_us * 1e-6 * speed_mm_s

    def residual_mm(self, accel_m_s2: float) -> float:
        """``a * T^2 / 24``, in mm -- the error mid-exposure makes under
        acceleration."""
        t_s = self.max_us * 1e-6
        return accel_m_s2 * t_s * t_s / 24.0 * 1000.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarise_sidecar(path: Path) -> CameraExposure:
    rows = read_frame_metadata_csv(path)
    if not rows:
        raise ValueError(f"{path.name}: no rows")
    camera = rows[0].camera
    exposures_us = [r.sensor_exposure_time_ns / 1000.0 for r in rows]
    reported = [e for e in exposures_us if e > 0.0]
    gains = [r.sensor_analog_gain for r in rows]
    median = statistics.median(reported) if reported else 0.0
    deviations = [abs(e - median) for e in reported]
    return CameraExposure(
        camera=camera,
        n_frames=len(rows),
        n_reported=len(reported),
        median_us=median,
        min_us=min(reported) if reported else 0.0,
        max_us=max(reported) if reported else 0.0,
        p95_abs_dev_us=_percentile(deviations, 95.0),
        gain_min=min(gains) if gains else 0.0,
        gain_max=max(gains) if gains else 0.0,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("episode_dir", type=Path)
    p.add_argument("--speed-mm-s", type=float, default=P95_HAND_SPEED_MM_S,
                   help="speed the millimetre figures are priced at; a timing error is "
                        "invisible at rest and linear in speed")
    p.add_argument("--accel-m-s2", type=float, default=TYPICAL_HAND_ACCEL_M_S2,
                   help="acceleration used for the intra-exposure second-order term")
    p.add_argument("--max-cross-camera-mm", type=float, default=0.50,
                   help="gate on the cross-camera sample-instant spread, priced at "
                        "--speed-mm-s. This is the part the per-frame correction "
                        "cannot remove, so it is the one real gate here")
    p.add_argument("--max-blur-mm", type=float, default=6.0,
                   help="advisory gate on motion blur (exposure * speed); it costs "
                        "detection precision, not timing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecars = sorted(args.episode_dir.glob(f"*.{SIDECAR_BASENAME}"))
    if not sidecars:
        print(
            f"cannot check: no *.{SIDECAR_BASENAME} under {args.episode_dir}",
            file=sys.stderr,
        )
        return 2

    summaries: list[CameraExposure] = []
    for path in sidecars:
        try:
            summaries.append(summarise_sidecar(path))
        except (ValueError, KeyError, OSError) as exc:
            print(f"cannot check: {exc}", file=sys.stderr)
            return 2

    silent = [s for s in summaries if not s.reported]
    if silent:
        print(
            "cannot check: no exposure reported by "
            + ", ".join(s.camera for s in silent)
            + ". The per-frame SOF -> mid-exposure correction is driven by this column, "
            "so an all-zero one does not mean 'no correction needed', it means the "
            "correction silently does nothing. Rebuild the recorders "
            "(check_recorder_build.py) and re-record.",
            file=sys.stderr,
        )
        return 2

    print(f"{'camera':<14}{'n':>7}{'median us':>12}{'span us':>10}"
          f"{'p95 dev us':>12}{'blur mm':>10}{'gain':>14}")
    for s in summaries:
        print(f"{s.camera:<14}{s.n_frames:>7}{s.median_us:>12.1f}{s.span_us:>10.1f}"
              f"{s.p95_abs_dev_us:>12.1f}{s.blur_mm(args.speed_mm_s):>10.2f}"
              f"{s.gain_min:>7.2f}-{s.gain_max:<6.2f}")

    medians = [s.median_us for s in summaries]
    cross_us = max(medians) - min(medians)
    cross_mm = cross_us * 0.5e-6 * args.speed_mm_s
    worst_blur = max(s.blur_mm(args.speed_mm_s) for s in summaries)
    worst_resid = max(s.residual_mm(args.accel_m_s2) for s in summaries)
    worst_wander = max(s.p95_abs_dev_us for s in summaries) * 0.5e-6 * args.speed_mm_s

    print(f"\nat {args.speed_mm_s:.0f} mm/s:")
    print(f"  within-camera wander   {worst_wander:6.2f} mm  "
          "-- CORRECTED per frame, shown only so the size is on record")
    print(f"  cross-camera spread    {cross_mm:6.2f} mm  "
          f"(gate {args.max_cross_camera_mm:.2f}) -- NOT corrected; one pose, one time")
    print(f"  motion blur            {worst_blur:6.2f} mm  "
          f"(gate {args.max_blur_mm:.2f}) -- precision, not timing")
    print(f"  intra-exposure 2nd ord {worst_resid:6.3f} mm  "
          f"at {args.accel_m_s2:.0f} m/s^2 -- the error in 'mid-exposure' itself")

    failures: list[str] = []
    if cross_mm > args.max_cross_camera_mm:
        failures.append(
            f"cameras sample {cross_mm:.2f} mm apart at {args.speed_mm_s:.0f} mm/s "
            f"({cross_us:.0f} us of exposure difference). Logging does not fix this: a "
            "fused pose has one timestamp, so the views disagree about when 'now' was. "
            "Equalise the exposures across cameras"
        )
    if worst_blur > args.max_blur_mm:
        failures.append(
            f"motion blur reaches {worst_blur:.2f} mm at {args.speed_mm_s:.0f} mm/s. "
            "That is a detection-precision cost, not a timing one -- shorten the "
            "exposure and raise the gain"
        )
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    print("\nOK: exposure is recorded, so the per-frame correction is live; what it "
          "cannot remove is inside the gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
