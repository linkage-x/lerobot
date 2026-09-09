#!/usr/bin/env python3
"""Open one already-enabled BOX gripper; never connect to/control the arm.

No mode change, zeroing, homing, automatic closing, or retry on failure.
Targets increase by at most 0.5 mm at no more than 10 Hz, each acknowledged
and checked against independently timestamped measured width before advancing.
"""
import argparse
import json
from pathlib import Path
import signal
import time


def open_gradually(box, target, events, *, clock=time.monotonic, sleep=time.sleep):
    if box.mode != 1:
        raise RuntimeError('Gripper must already be in control mode; will not switch modes')
    initial = []
    for _ in range(20):
        seq, width = box.wait_fresh(timeout=.2)
        initial.append(width)
        events.append(dict(kind='initial', time_s=clock(), sequence=seq, width_m=width))
    if max(initial) - min(initial) > .0002:
        raise RuntimeError('Initial gripper width is not stable')
    if target < width - .0001 or not 0 <= target <= .0887398049235344:
        raise ValueError('Only opening within verified width range is allowed')
    first, commanded, count, last_sent = width, width, 0, -float('inf')
    while target - commanded > 1e-9:
        rc, mode = box.box.get_mode(box.device_id, timeout_ms=500)
        if rc != 0 or mode != 1:
            raise RuntimeError('Control-mode readback changed; no mode switching')
        seq, width = box.wait_fresh(timeout=.2)
        if abs(width - commanded) > .001:
            raise RuntimeError('Width moved away from previous target')
        next_target = min(target, commanded + .0005)
        sleep(max(0., .1 - (clock() - last_sent)))
        if box.send(next_target) is not True:
            raise RuntimeError('Opening command not acknowledged; no retry')
        last_sent = clock()
        events.append(dict(kind='submitted', time_s=last_sent, target_m=next_target))
        count += 1
        deadline, matched = last_sent + 2., 0
        while clock() < deadline:
            seq, width = box.wait_fresh(timeout=.2)
            events.append(dict(kind='feedback', time_s=clock(), sequence=seq, width_m=width))
            if width < first - .001 or width > next_target + .002:
                raise RuntimeError('Unexpected closing or opening overshoot')
            matched = matched + 1 if abs(width - next_target) <= .00025 else 0
            if matched >= 3:
                break
        else:
            raise RuntimeError('Measured gripper did not reach incremental target')
        commanded = next_target
        if count % 20 == 0:
            print(f'开度反馈 {width*1000:.2f} mm，目标 {target*1000:.2f} mm', flush=True)
    end = clock() + .5
    while clock() < end:
        seq, width = box.wait_fresh(timeout=.2)
        events.append(dict(kind='final_feedback', time_s=clock(), sequence=seq, width_m=width))
        if abs(width - target) > .0005:
            raise RuntimeError('Final opening did not remain at target')
    return dict(completed=True, initial_width_m=first, target_width_m=target,
                final_measured_width_m=width, submitted_commands=count,
                mode_changes_sent=0, arm_commands_sent=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-mm', type=float, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required')
    if not 0 <= a.target_mm <= 88.7398049235344:
        p.error('Target outside verified width range')
    import sys
    root = a.bundle.resolve()
    sys.path.insert(0, str(root))
    from runtime import verify_bundle, save
    from box_transport import BoxTransport
    import fcntl
    verify_bundle(root)
    config = json.loads((root / 'config.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    events = []
    outcome = dict(completed=False, arm_commands_sent=0, mode_changes_sent=0)
    def abort(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}; no further width commands')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, abort)
    with (root / 'session.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        box = BoxTransport(config['box_sdk_dir'], config['gripper_device_id'], config['gripper_ip'], allow_commands=True)
        try:
            box.open()
            print(f'夹爪 ID={box.device_id}，模式={box.mode}；仅打开夹爪，不连接机械臂。', flush=True)
            outcome.update(open_gradually(box, a.target_mm / 1000., events))
        except BaseException as exc:
            outcome['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            box.close()
            save(a.output / 'events.json', events)
            save(a.output / 'result.json', outcome)
    print(json.dumps(outcome, indent=2), flush=True)
    return 0 if outcome['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
