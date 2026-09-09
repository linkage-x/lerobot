#!/usr/bin/env python3
"""No device access: pure planner, native evaluator, synthetic-state fault tests."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from replay_ik_trajectory_guarded import require_candidate, sha256
from replay_timed_candidate import FreshBoxMeasurement, GripperClock, dispatch_gripper, evaluate, fk


def run(plan, core):
    require_candidate(core)
    assert core._FR3_TIMED_OFFLINE_CANDIDATE == 1
    chain = plan["chain"]
    tests, results = [], []

    def new(ep):
        return core.TimedReplayCandidate(ep["times_s"], ep["coeff_descending_unit_interval"],
                                         chain["origins"], chain["axes"], chain["tail"])

    def begin(ep):
        c = new(ep)
        target = c.evaluate(0.)
        c.feed_gripper(1, 0., float(target[7]))
        c._offline_start(target[:7], np.zeros(7), c.fk(target[:7]))
        return c

    def expect_fault(name, ep, mutate, code, ticks=1000):
        c = begin(ep)
        caught = False
        for i in range(1, ticks+1):
            t = i * .001
            target = c.evaluate(t)
            q, dq, tcp = target[:7].copy(), c.evaluate(t, 1)[:7], c.fk(target[:7])
            c._offline_set_time(t)
            if name != "stale_gripper":
                c.feed_gripper(i+1, t, float(target[7]) if name != "gripper_tracking" else .06)
            t, q, dq, tcp = mutate(c, t, q, dq, tcp)
            try:
                c._offline_step(t, q, dq, tcp, 1)
            except RuntimeError:
                assert c.fault_code() == code, (name, c.fault_code(), code)
                caught = True
                try:
                    c._offline_step(t+.001, target[:7], np.zeros(7), c.fk(target[:7]), 1)
                except RuntimeError:
                    break
                raise AssertionError("Fault did not latch")
        assert caught, name
        tests.append(dict(name=name, fault_code=code, stopped_at_s=t, latched=True))

    rng = np.random.default_rng(13)
    for ep0 in plan["episodes"]:
        ep = dict(ep0, times_s=np.asarray(ep0["times_s"]),
                  coeff_descending_unit_interval=np.asarray(ep0["coeff_descending_unit_interval"]))
        c = new(ep)
        maximum_error = np.zeros(4)
        maximum_fk_error = 0.
        query_times = np.r_[ep["times_s"], rng.uniform(0., c.duration(), 1500), c.duration()+1.]
        for t in query_times:
            for derivative in range(4):
                error = np.max(abs(c.evaluate(float(t), derivative) - evaluate(ep, float(t), derivative)))
                maximum_error[derivative] = max(maximum_error[derivative], error)
            q = c.evaluate(float(t))[:7]
            maximum_fk_error = max(maximum_fk_error, np.max(abs(c.fk(q)-fk(chain, q))))
        assert max(maximum_error) < 1e-8 and maximum_fk_error < 1e-10

        # Full-duration synthetic perfect tracking at 1 kHz, not a real robot.
        c = begin(ep)
        scheduler = GripperClock(ep)
        sent = []
        finished = False
        for i in range(int(np.ceil((c.duration()+.2)/.001))+1):
            t = i*.001
            desired = c.evaluate(t)
            c._offline_set_time(t)
            if i and i % 20 == 0:
                c.feed_gripper(i+1, t, float(desired[7]))
            if i % 5 == 0:
                command = dispatch_gripper(c, scheduler, lambda width: True)
                if command is not None:
                    sent.append((t, command))
                    assert abs(command-desired[7]) < 1e-12
            out = c._offline_step(t, desired[:7], c.evaluate(t, 1)[:7], c.fk(desired[:7]), 1)
            if out["finished"]:
                finished = True
                break
        assert finished and c.fault_code() == 0
        assert min(np.diff([s[0] for s in sent])) >= 1/15. - 1e-12
        results.append(dict(episode=ep["episode"], synthetic_ticks=i+1, duration_s=c.duration(),
                            cpp_python_maximum_error_by_derivative=maximum_error.tolist(),
                            cpp_python_maximum_fk_matrix_error=float(maximum_fk_error),
                            synthetic_gripper_commands=len(sent), completed=True))
        print(f"Synthetic Episode {ep['episode']}: PASS, {i+1} ticks, {len(sent)} commands", flush=True)

    ep = plan["episodes"][0]
    same = lambda c,t,q,dq,tcp: (t,q,dq,tcp)
    expect_fault("stale_gripper", ep, same, 7)
    expect_fault("external_transport_failure", ep,
                 lambda c,t,q,dq,tcp: (c.abort_external() or t,q,dq,tcp), 6)
    expect_fault("clock_gap", ep, lambda c,t,q,dq,tcp: (t+.02,q,dq,tcp), 5)
    def wall(c,t,q,dq,tcp):
        q[5] = 4.50
        return t,q,dq,c.fk(q)
    expect_fault("wall_entry", ep, wall, 9)
    def bad_tcp(c,t,q,dq,tcp):
        tcp[2,3] += .01
        return t,q,dq,tcp
    expect_fault("wrong_tcp_configuration", ep, bad_tcp, 2)
    def track(c,t,q,dq,tcp):
        q[6] += .06
        return t,q,dq,c.fk(q)
    expect_fault("joint_tracking", ep, track, 11)
    def cartesian(c,t,q,dq,tcp):
        q[0] += .04
        return t,q,dq,c.fk(q)
    expect_fault("tcp_tracking", ep, cartesian, 12)
    expect_fault("gripper_tracking", ep, same, 13)
    def nan_q(c,t,q,dq,tcp):
        q[0] = np.nan
        return t,q,dq,tcp
    expect_fault("nan_state", ep, nan_q, 8)

    c = begin(ep)
    try:
        dispatch_gripper(c, GripperClock(ep), lambda width: False)
    except RuntimeError:
        try:
            c._offline_step(.001,c.evaluate(.001)[:7],np.zeros(7),c.fk(c.evaluate(.001)[:7]),1)
        except RuntimeError:
            assert c.fault_code() == 6
            tests.append(dict(name="unacknowledged_gripper_command_aborts_arm", passed=True))
        else:
            raise AssertionError("Missing ACK did not stop native controller")
    else:
        raise AssertionError("Missing ACK accepted")
    c = begin(ep)
    for name, args in [("duplicate_measurement", (1, 0., .05)),
                       ("future_measurement", (2, .01, .05)),
                       ("invalid_width", (2, 0., .1))]:
        try:
            c.feed_gripper(*args)
        except ValueError:
            tests.append(dict(name=name, rejected=True))
        else:
            raise AssertionError(name)
    c = new(ep)
    q = c.evaluate(0.)[:7]
    c.feed_gripper(1,0.,float(c.evaluate(0.)[7]))
    bad_q = q.copy(); bad_q[0] += .02
    try:
        c._offline_start(bad_q,np.zeros(7),c.fk(bad_q))
    except RuntimeError:
        assert c.fault_code() == 1
        tests.append(dict(name="wrong_first_pose", rejected=True))
    else:
        raise AssertionError("Incorrect start was accepted")

    scheduler = GripperClock(ep)
    first = scheduler.command(0.)
    # Delaying start in wall time cannot advance either target clock.
    assert first == evaluate(ep,0.)[7] and scheduler.command(0.) is None
    try:
        scheduler.command(-.001)
    except ValueError:
        tests.append(dict(name="invalid_gripper_clock", rejected=True))
    else:
        raise AssertionError("Invalid clock accepted")
    endpoint = GripperClock(ep)
    endpoint.command(0.)
    assert endpoint.command(ep["duration_s"]) == evaluate(ep,ep["duration_s"])[7]
    assert endpoint.command(ep["duration_s"]+1.) is None
    tests.append(dict(name="gripper_final_command_once", passed=True))
    decoder = FreshBoxMeasurement(123)
    snap = dict(status=dict(active=True, device_id=123),
                sensors=dict(box_gripper=dict(timestamp=100, distance_m=.05)))
    assert decoder.decode(snap) is None
    # An SDK cache marked 'fresh' must not keep the native watchdog alive.
    for _ in range(200):
        assert decoder.decode(snap) is None
    snap["sensors"]["box_gripper"]["timestamp"] = 101
    assert decoder.decode(snap) == (1, .05)
    tests.append(dict(name="cached_sensor_cannot_refresh_watchdog", passed=True))
    for name, mutation in [("wrong_gripper_device", lambda s: s["status"].update(device_id=456)),
                           ("backwards_MCU_timestamp", lambda s: s["sensors"]["box_gripper"].update(timestamp=99)),
                           ("missing_MCU_timestamp", lambda s: s["sensors"]["box_gripper"].update(timestamp=None))]:
        copy = json.loads(json.dumps(snap))
        mutation(copy)
        try:
            decoder.decode(copy)
        except ValueError:
            tests.append(dict(name=name, rejected=True))
        else:
            raise AssertionError(name)
    for option in ("--execute", "--start-only"):
        p = subprocess.run([sys.executable, str(Path(__file__).with_name("replay_timed_candidate.py")), option], capture_output=True, text=True)
        assert p.returncode == 9 and "BLOCKED" in p.stderr
        tests.append(dict(name=option+" blocked before any device access", passed=True))
    return dict(hardware_ready=False, robot_instances_created=0, physical_commands_sent=0,
                native_core_path=core.__file__, native_core_sha256=sha256(core.__file__),
                episodes=results, fault_and_interface_tests=tests,
                limitation="Synthetic states validate software branches, not tracking dynamics, stop distance or timing guarantees on real hardware.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from panda_py import _core  # Import only. Never construct Panda or Robot.
    result = run(json.loads(args.plan.read_text()), _core)
    result["plan_sha256"] = sha256(args.plan)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(f"PASS: {len(result['fault_and_interface_tests'])} fault/interface checks; hardware remains blocked.")
