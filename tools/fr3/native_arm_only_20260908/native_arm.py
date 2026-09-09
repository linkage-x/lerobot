"""Latest calibrated IK, original panda-py JointTrajectory, passive gripper only.

The bound controller is the unchanged C++ class used by move_to_joint_position.
Only lifecycle plumbing differs: execute the audited object, no auto recovery.
No BOX imports, connection, mode changes or gripper commands.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/nvidia/box_api/replay_p0_once_20260908')
ARM = Path('/home/nvidia/box_api/replay_p0_arm_only_20260908')
AUDIT = Path('/home/nvidia/box_api/replay_p0_native_reuse_20260908')
sys.path[:0] = [str(ROOT/'python'), str(ARM), str(AUDIT), str(AUDIT/'build')]
from benchmark_native import knots, chunk_indices, SOURCE_SHA, digest
from arm_runtime import snapshot, state_checks, verify
from replay_timed_candidate import urdf_chain, WIDTH_MAX, fk
from replay_ik_trajectory_guarded import LOWER, UPPER, WALL_BAND
import plans

STIFFNESS = [300., 300., 300., 300., 120., 80., 30.]
DAMPING = [25., 25., 25., 25., 10., 8., 5.]


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def verify_inputs():
    verify(ARM)
    if digest(SOURCE/'timed_plan.json') != SOURCE_SHA:
        raise RuntimeError('Calibration / IK input changed')
    for name, expected in json.loads((ROOT/'manifest.json').read_text())['files'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f'Changed native artifact: {name}')
    for name, expected in json.loads((ROOT/'manifest.json').read_text())['external_files'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Changed audit dependency: {name}')


def prepare(core, q, speed, width, out):
    """Original planner plus independent derivative and full-mesh audit."""
    import _native_plan_audit as audit
    q = np.asarray(q, float)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Invalid waypoints')
    if np.any(np.linalg.norm(np.diff(q, axis=0), axis=1) <= 1e-10):
        raise ValueError('Duplicate native waypoints; not silently dropped')
    trajectory = core.JointTrajectory(q.tolist(), speed, .02, 30.)
    independent = audit.Trajectory(audit.Path(q.tolist(), .02),
        np.array([2.175]*4+[2.610]*3)*speed,
        np.array([15,7.5,10,12.5,15,20,20])*speed, .001)
    duration = trajectory.get_duration()
    if not independent.valid() or not 0 < duration < 600 or abs(duration-independent.duration()) > 1e-8:
        raise RuntimeError('Native planning failed / audit mismatch')
    t = np.linspace(0, duration, int(np.ceil(duration/.001))+1)
    data = np.array([independent.state(float(x)) for x in t])
    if not np.isfinite(data).all():
        raise RuntimeError('Nonfinite native output')
    check = np.unique(np.r_[np.arange(0,len(t),20),len(t)-1])
    direct = np.array([trajectory.get_joint_positions(float(t[i])) for i in check])
    if np.max(abs(direct-data[check,:7])) > 1e-9:
        raise RuntimeError('Audited and executable paths differ')
    wall = float(np.minimum(data[:,:7]-LOWER-WALL_BAND, UPPER-WALL_BAND-data[:,:7]).min())
    velocity = abs(data[:,7:14]).max(axis=0)
    acceleration = abs(data[:,14:21]).max(axis=0)
    jerk_sampled = abs(np.diff(data[:,14:21],axis=0)/np.diff(t)[:,None]).max(axis=0)
    # The legacy six-episode preflight's 80% FR3 limits, not custom pilot limits.
    limit_v = np.array([1.6768]*4+[3.3664,2.6752,3.3664])
    report = dict(duration_s=duration, native_speed_factor=speed,
        native_max_deviation_rad=.02, minimum_wall_margin_rad=wall,
        velocity_peak_rad_s=velocity.tolist(), acceleration_peak_rad_s2=acceleration.tolist(),
        sampled_jerk_peak_rad_s3=jerk_sampled.tolist(),
        jerk_continuity_certified=False, gripper_control=False, declared_width_m=width)
    np.savez_compressed(out/'native_plan.npz', t=t, data=data, source_waypoints=q)
    if wall <= 0 or np.any(velocity>limit_v) or np.any(acceleration>6.4) or np.any(jerk_sampled>3200):
        save(out/'audit_failed.json',report)
        raise RuntimeError('Native trajectory exceeds legacy joint-motion envelope')
    original = plans.samples
    try:
        plans.samples = lambda _: ((float(t[i]),data[i,:7],width) for i in check)
        report['geometry'] = plans.geometry({'episode':out.name}, ARM/'model.urdf',
                                             json.loads((ARM/'scene.json').read_text()))
    finally:
        plans.samples = original
    save(out/'audit.json',report)
    if not report['geometry']['sampled_geometry_pass']:
        raise RuntimeError('Native passive-gripper / full-link geometry failed')
    print(f'{out.name}: native plan {duration:.1f}s, sampled geometry passed',flush=True)
    return trajectory


def execute(panda, core, trajectory, target, chain, out):
    controller = core.NativeJointTrajectoryController(trajectory, STIFFNESS, DAMPING, .001)
    rows = []
    started = False
    result = dict(completed=False, gripper_commands_sent=0, gripper_sdk_loaded=False)
    try:
        panda.start_controller_guarded(controller)
        started = True
        deadline = time.monotonic()+trajectory.get_duration()+30
        while panda.control_thread_active():
            s = panda.get_state()
            rows.append([controller.get_time(),*s.q,*s.dq,*s.O_T_EE,
                         s.control_command_success_rate,*s.tau_ext_hat_filtered])
            if time.monotonic()>deadline:
                raise RuntimeError('Native controller timeout; no automatic continuation')
            time.sleep(.02)
        panda.stop_controller()
        panda.raise_error()
        s = panda.get_state()
        error = float(abs(np.asarray(s.q)-target).max())
        result['endpoint_max_error_rad'] = error
        # Original native success test is Eigen relative isApprox (precision .01).
        relative_ok = np.linalg.norm(np.asarray(s.q)-target) <= .01*min(np.linalg.norm(s.q),np.linalg.norm(target))
        if controller.get_time()<trajectory.get_duration() or not relative_ok or error>.03:
            raise RuntimeError('Original native completion / endpoint check failed')
        target_tcp, measured_tcp = fk(chain,target), fk(chain,np.asarray(s.q))
        position_error = float(np.linalg.norm(target_tcp[:3,3]-measured_tcp[:3,3]))
        angle_error = float(np.rad2deg(np.arccos(np.clip((np.trace(target_tcp[:3,:3].T@measured_tcp[:3,:3])-1)/2,-1,1))))
        result.update(tcp_position_error_m=position_error,tcp_orientation_error_deg=angle_error)
        if position_error>.03 or angle_error>8:
            raise RuntimeError('Original native endpoint TCP error exceeded')
        result['completed'] = True
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if started:
            panda.stop_controller()
        result['motion_started'] = started
        result['controller_time_s'] = controller.get_time()
        np.savez_compressed(out/'telemetry.npz',data=np.asarray(rows),
            columns='time,q[7],dq[7],O_T_EE_column_major[16],command_success_rate,tau_ext[7]')
        save(out/'execution.json',result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=int,choices=[0,1])
    p.add_argument('--execute',action='store_true')
    p.add_argument('--gripper-width-mm',type=float,required=True)
    a = p.parse_args()
    width = a.gripper_width_mm/1000
    if not np.isfinite(width) or not 0<=width<=WIDTH_MAX:
        p.error('Invalid measured passive gripper width')
    verify_inputs()
    from panda_py import _core as core, libfranka
    source = json.loads((SOURCE/'timed_plan.json').read_text())
    ep = next(e for e in source['episodes'] if e['episode']==a.episode)
    q = knots(ep)[:,:7]
    chain = urdf_chain(ARM/'model.urdf')
    if chain != source['chain']:
        raise RuntimeError('URDF / calibration chain mismatch')
    reference = json.loads((ARM/'tool_reference.json').read_text())
    out = ROOT/'logs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    print(f'日志目录：{out}',flush=True)
    if a.execute:
        if os.geteuid()!=0:
            p.error('Use sudo for the original 1 ms FCI network test; no kernel settings are changed')
        if not sys.stdin.isatty() or input('夹爪空夹且开口保持不变、范围清空、急停可用，输入 YES 执行只动臂重播：').strip()!='YES':
            print('Cancelled: no motion');return 2
    with ExitStack() as stack:
        for root in ([ROOT,ARM,SOURCE] if a.execute else []):
            lock = stack.enter_context((root/'session.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Offline nominal audit before any robot connection or motion.
        for k,(i,j) in enumerate(chunk_indices(len(q),200)):
            folder=out/f'nominal_{k}';folder.mkdir()
            prepare(core,q[i:j],.01,width,folder)
        if not a.execute:
            print('Offline check finished. Robot instances=0; gripper instances=0.');return 0
        probe=subprocess.run(['ping','-I','192.168.11.100','-i','0.001','-c','10000','-s','1200','-q','192.168.11.102'],
                             text=True,capture_output=True,timeout=25)
        save(out/'network.json',dict(stdout=probe.stdout,stderr=probe.stderr,returncode=probe.returncode))
        loss=re.search(r'([\d.]+)% packet loss',probe.stdout)
        rtt=re.search(r'= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms',probe.stdout)
        if probe.returncode or not loss or not rtt or float(loss[1])!=0 or float(rtt[3])>.90 or float(rtt[4])>.10:
            raise RuntimeError('Original FCI network gate failed; no motion')
        panda=core.Panda('192.168.11.102','native_arm_only',libfranka.RealtimeConfig.kIgnore)
        def interrupt(sig, frame):
            raise KeyboardInterrupt(f'Signal {sig}')
        signal.signal(signal.SIGTERM,interrupt)
        # Native start and chunks are planned from the measured position, as in
        # the old API. Audit each actual object; execute that same object.
        stages=[('start',q[:1],.03)]+[(f'replay_{k}',q[i+1:j],.01) for k,(i,j) in enumerate(chunk_indices(len(q),200))]
        try:
            for name,targets,speed in stages:
                s=snapshot(panda.get_robot());state_checks(s,chain,reference)
                if name=='start' and abs(np.asarray(s['q'])-q[0]).max()<=.03:
                    print('Already within original start tolerance; no start movement.');continue
                folder=out/name;folder.mkdir()
                path=np.vstack([s['q'],targets])
                trajectory=prepare(core,path,speed,width,folder)
                state_checks(snapshot(panda.get_robot()),chain,reference,s['q'])
                save(folder/'before.json',s)
                execute(panda,core,trajectory,targets[-1],chain,folder)
                state_checks(snapshot(panda.get_robot()),chain,reference)
        finally:
            panda.stop_controller()
        print('Completed arm-only native replay. No gripper commands.',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
