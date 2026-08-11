#!/usr/bin/env python

from __future__ import annotations

import argparse
import time

import numpy as np

from lerobot.teleoperators.spacemouse.backend import PySpaceMouseDriver


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print raw SpaceMouse readings without teleop-side filtering or bias removal.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--print-header-every", type=int, default=50)
    return parser.parse_args(argv)


def _format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value): .5f}" for value in values)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    driver = PySpaceMouseDriver(device_id=int(args.device_id))
    driver.connect()

    interval_s = 0.0 if args.fps <= 0 else 1.0 / float(args.fps)
    deadline = None if args.duration_s is None else time.perf_counter() + float(args.duration_s)
    header_every = max(int(args.print_header_every), 1)
    sample_idx = 0

    print("Connected to SpaceMouse.")
    print(f"Device: {driver.describe()}")
    print("Columns:")
    print("  idx t_s tx ty tz roll pitch yaw trans_norm rot_norm buttons")

    try:
        start = time.perf_counter()
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            loop_start = time.perf_counter()
            reading = driver.poll()
            if reading is None:
                print(f"{sample_idx:06d} {time.perf_counter() - start:8.3f} poll=None")
            else:
                translation = np.asarray(reading.translation, dtype=np.float64)
                rotation = np.asarray(reading.rotation, dtype=np.float64)
                trans_norm = float(np.linalg.norm(translation))
                rot_norm = float(np.linalg.norm(rotation))
                if sample_idx % header_every == 0 and sample_idx != 0:
                    print("  idx t_s tx ty tz roll pitch yaw trans_norm rot_norm buttons")
                print(
                    f"{sample_idx:06d} "
                    f"{time.perf_counter() - start:8.3f} "
                    f"{_format_vec(translation)} "
                    f"{_format_vec(rotation)} "
                    f"{trans_norm: .5f} "
                    f"{rot_norm: .5f} "
                    f"{reading.buttons}"
                )

            sample_idx += 1
            sleep_s = interval_s - (time.perf_counter() - loop_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    finally:
        driver.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
