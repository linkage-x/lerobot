#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Drive the FR3 back and forth by a small joint offset, sampling FCI link health.

Exists to give the 1 kHz control loop a repeatable workload: `move_to_start` is
useless as a probe once the arm is already at the start pose. Each leg samples
`control_command_success_rate` -- the robot's own count of how many of the last
100 command packets arrived in time -- so a topology or cabling change can be
A/B'd without guessing from a single abort message.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import threading
import time

DEFAULT_ROBOT_IP = "192.168.11.102"
# Wrist and elbow: visible motion, small swept volume, no risk of hitting the base.
DEFAULT_JOINTS = "3,5,6"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cycle the FR3 by a small joint offset and report FCI link health.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    parser.add_argument("--cycles", type=int, default=5, help="Number of out-and-back cycles.")
    parser.add_argument("--amplitude", type=float, default=0.15, help="Joint offset in radians (default 0.15 rad ~ 8.6 deg).")
    parser.add_argument("--joints", default=DEFAULT_JOINTS, help="Comma-separated joint indices (0-6) to move.")
    parser.add_argument("--speed-factor", type=float, default=0.15, help="panda_py speed factor (0-1).")
    parser.add_argument("--settle-s", type=float, default=0.3, help="Pause between legs.")
    return parser.parse_args(argv)


class RateSampler:
    """Polls control_command_success_rate while a blocking motion runs."""

    def __init__(self, panda, period_s: float = 0.02) -> None:
        self._panda = panda
        self._period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[float] = []

    def __enter__(self) -> RateSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rate = self._panda.get_state().control_command_success_rate
            except Exception:
                rate = None
            # 0.0 is what an idle robot reports; it would drag the summary down.
            if rate:
                self.samples.append(rate)
            self._stop.wait(self._period_s)


def active_errors(errors) -> list[str]:
    return [name for name in dir(errors) if not name.startswith("_") and getattr(errors, name) is True]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    joints = [int(tok) for tok in args.joints.split(",") if tok.strip()]
    if any(j < 0 or j > 6 for j in joints):
        raise SystemExit(f"joint indices must be in 0..6, got {joints}")

    import numpy as np
    from panda_py import Panda

    panda = Panda(args.robot_ip)
    home = np.array(panda.get_state().q, dtype=float)
    target = home.copy()
    for j in joints:
        target[j] += args.amplitude

    print(f"fr3_cycle_motion=START joints={joints} amplitude={args.amplitude}rad "
          f"cycles={args.cycles} speed_factor={args.speed_factor}")
    print(f"  home q = {[round(v, 4) for v in home]}")

    all_rates: list[float] = []
    aborts = 0
    legs = 0
    try:
        for cycle in range(args.cycles):
            for label, goal in (("out", target), ("back", home)):
                legs += 1
                with RateSampler(panda) as sampler:
                    t0 = time.perf_counter()
                    panda.move_to_joint_position(goal, speed_factor=args.speed_factor)
                    dt = time.perf_counter() - t0
                errs = active_errors(panda.get_state().last_motion_errors)
                if errs:
                    aborts += 1
                rates = sampler.samples
                all_rates.extend(rates)
                summary = (
                    f"min={min(rates):.3f} mean={statistics.mean(rates):.3f} max={max(rates):.3f} n={len(rates)}"
                    if rates
                    else "no samples"
                )
                print(f"  cycle {cycle} {label:<4} {dt:5.2f}s  success_rate {summary}"
                      f"{'  ABORT=' + ','.join(errs) if errs else ''}")
                time.sleep(args.settle_s)
    finally:
        with contextlib.suppress(Exception):
            panda.stop_controller()

    print(f"fr3_cycle_motion={'FAIL' if aborts else 'PASS'} legs={legs} aborted={aborts}")
    if all_rates:
        ordered = sorted(all_rates)
        print(f"  control_command_success_rate over {len(all_rates)} samples: "
              f"p05={ordered[int(0.05 * len(ordered))]:.3f} "
              f"p50={statistics.median(ordered):.3f} "
              f"mean={statistics.mean(ordered):.3f} "
              f"max={ordered[-1]:.3f}")
        print("  1.000 = every command packet arrived in time; 0.70 = 30% missed the 1 ms deadline.")
    return 1 if aborts else 0


if __name__ == "__main__":
    raise SystemExit(main())
