#!/usr/bin/env python3
"""Check that an episode's cameras actually held their exposure, and price the drift.

The frame label is start-of-frame, but the scene was sampled around
mid-exposure.  So every frame's position label carries an offset of roughly
``exposure / 2``.  A *constant* offset is harmless -- T3c measures it once and
it comes out in the wash.  A *varying* one does not, and under Argus
auto-exposure it varies with scene brightness, which varies with pose.  That
makes it a pose-correlated timing bias: it never averages out, and no amount of
clock work can find it, because both clocks are perfectly fine.

    label error (mm) = (exposure - median exposure) / 2 * |v|

At 680 mm/s -- this rig's p95 hand speed -- an exposure swinging between 4 ms
and 12 ms is about 2.7 mm of wander in the position label alone, comparable to
the whole 3 mm being adjudicated.  Hence this script: locking the exposure is
cheap, and *verifying* it stayed locked is cheaper still, but only if somebody
looks.

    python -m tools.thor.gmsl2.check_exposure_lock <episode_dir>
    python -m tools.thor.gmsl2.check_exposure_lock <episode_dir> --speed-mm-s 1500

Exit codes: 0 locked within tolerance, 1 it drifted (a finding about the
recording), 2 it could not be checked (a finding about the setup) -- the same
0/1/2 split the laser-tracker CLIs use.
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
        populate it, both show up as all-zero -- and reporting that as
        "perfectly locked" would be exactly backwards.
        """
        return self.n_reported > 0 and self.median_us > 0.0

    def label_wander_mm(self, speed_mm_s: float) -> float:
        """p95 of the *varying* part of the SOF -> mid-exposure offset.

        Half the exposure deviation, because mid-exposure moves by half of what
        the integration window does.  The median is subtracted on purpose: the
        constant part is what T3c calibrates out, so charging it here would
        double-count it.
        """
        return self.p95_abs_dev_us * 0.5e-6 * speed_mm_s


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
                   help="speed the label wander is priced at; a timing error is "
                        "invisible at rest and linear in speed")
    p.add_argument("--max-wander-mm", type=float, default=0.10,
                   help="per-camera gate on the varying part of the SOF -> mid-exposure "
                        "offset. Default is a twentieth of the 3 mm under test")
    p.add_argument("--max-cross-camera-us", type=float, default=200.0,
                   help="gate on the spread of median exposure across cameras; they "
                        "label different instants if this is large")
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
            + ". Either the sidecar predates the sensor_exposure_time_ns column or the "
            "driver does not populate it -- an all-zero column is not a locked exposure.",
            file=sys.stderr,
        )
        return 2

    print(f"{'camera':<14}{'n':>7}{'median us':>12}{'span us':>10}"
          f"{'p95 dev us':>12}{'wander mm':>11}{'gain':>14}")
    worst = 0.0
    for s in summaries:
        wander = s.label_wander_mm(args.speed_mm_s)
        worst = max(worst, wander)
        print(f"{s.camera:<14}{s.n_frames:>7}{s.median_us:>12.1f}{s.span_us:>10.1f}"
              f"{s.p95_abs_dev_us:>12.1f}{wander:>11.3f}"
              f"{s.gain_min:>7.2f}-{s.gain_max:<6.2f}")

    medians = [s.median_us for s in summaries]
    cross_us = max(medians) - min(medians)
    print(f"\nat {args.speed_mm_s:.0f} mm/s: worst per-camera label wander "
          f"{worst:.3f} mm (gate {args.max_wander_mm:.3f})")
    print(f"cross-camera median exposure spread {cross_us:.1f} us "
          f"(gate {args.max_cross_camera_us:.1f}) -> "
          f"{cross_us * 0.5e-6 * args.speed_mm_s:.3f} mm of relative label offset")

    failures: list[str] = []
    if worst > args.max_wander_mm:
        failures.append(
            f"exposure moved during the episode: {worst:.3f} mm of label wander at "
            f"{args.speed_mm_s:.0f} mm/s. Set cameras.exposure_us to pin it -- this bias "
            "is pose-correlated, so it does not average out and the clock chain cannot see it"
        )
    if cross_us > args.max_cross_camera_us:
        failures.append(
            f"cameras hold different exposures ({cross_us:.1f} us apart), so they label "
            "different instants; a multi-camera solve then mixes times as well as views"
        )
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    print("OK: exposure held, and the residual wander is inside the gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
