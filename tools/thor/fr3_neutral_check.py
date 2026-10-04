"""Explicit physical test: home FR3, then hold with zero SpaceMouse increments.

This uses the production native worker and its guards. It does not open a mouse
or BOX and cannot forward operator input. Cameras can stay connected on Thor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import socket
import statistics
import subprocess
import time

from tools.thor.fr3_control_worker import DELTA_KEYS, load_config
from tools.thor.fr3_teleop import ThorFr3Session


def scheduling(pid):
    result = []
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            fields = (task / "stat").read_text().rsplit(") ", 1)[1].split()
            result.append({"tid": int(task.name), "name": (task / "comm").read_text().strip(),
                           "rt_priority": int(fields[37]), "policy": int(fields[38])})
        except (OSError, ValueError, IndexError):
            continue
    return result


def run(config_path, seconds):
    config = load_config(str(config_path))
    session = ThorFr3Session(config, config_path, Path(__file__).resolve().parents[2], None, emit=lambda _: None)
    process, channel = session._spawn_worker()
    rates, states, threads = [], [], []
    started = None
    startup = time.monotonic()
    failure = ""
    try:
        while True:
            tick = time.monotonic()
            for _ in range(16):
                if b"\n" not in channel.buffer and not select.select([channel.sock], [], [], 0)[0]:
                    break
                message = channel.receive()
                if message.get("state") in ("error", "stopped"):
                    raise RuntimeError(message.get("message") or "Native worker stopped")
                state = message.get("telemetry") or {}
                if message.get("state") == "running" and state:
                    if started is None:
                        started = time.monotonic()
                    rates.append(state["control_command_success_rate"])
                    states.append(state)
                    if not threads:
                        threads = scheduling(process.pid)
            if started is not None:
                if tick - started >= seconds:
                    break
                channel.send({"op": "action", "sent_monotonic_s": time.monotonic(),
                              "action": {"enabled": False, **dict.fromkeys(DELTA_KEYS, 0.), "gripper": .5}})
            else:
                if tick - startup > 60:
                    raise TimeoutError("Neutral test startup timed out")
                channel.send({"op": "heartbeat", "sent_monotonic_s": time.monotonic()})
            if process.poll() is not None:
                raise RuntimeError(f"Native worker exited with {process.returncode}")
            time.sleep(max(0, .01 - (time.monotonic() - tick)))
    except Exception as exc:
        failure = str(exc)
    finally:
        try:
            channel.send({"op": "stop"})
        except (OSError, ValueError):
            pass
        channel.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    return {"passed": not failure and bool(rates), "error": failure,
            "hold_seconds": 0 if started is None else time.monotonic() - started,
            "samples": len(rates), "min_success_rate": min(rates) if rates else None,
            "mean_success_rate": statistics.mean(rates) if rates else None,
            "below_target_samples": sum(rate < config["fr3_teleop"]["min_success_rate"] for rate in rates),
            "native_threads": threads, "worker_exited": process.poll() is not None,
            "telemetry": states}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-motion", action="store_true", help="Required: acknowledge physical homing")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--config-path", type=Path, default=Path("tools/thor/gmsl2/thor_fr3_teleop.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_motion or not 1 <= args.seconds <= 60:
        parser.error("--confirm-motion and a hold duration in [1, 60] seconds are required")
    result = run(args.config_path.resolve(), args.seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("telemetry", "native_threads")}, indent=2))
    print("Native FIFO threads:", [t for t in result["native_threads"] if t["policy"] == 1])
    raise SystemExit(0 if result["passed"] else 1)
