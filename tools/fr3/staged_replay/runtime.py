#!/usr/bin/env python3
"""Staged replay: explicit read-only preparation, separate motion authorization.

The runner never clears Reflex/UserStopped, changes tool/collision parameters,
modifies Linux tuning, or falls back to the old native waypoint planner.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from replay_timed_candidate import GripperClock, dispatch_gripper, evaluate, fk, urdf_chain
from replay_ik_trajectory_guarded import require_candidate
from box_transport import BoxTransport
from plans import geometry, transit, start_check, validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    # Validate serialization before creating the final file, so a type error
    # cannot leave a truncated report masquerading as a completed diagnostic.
    text=json.dumps(data,indent=2,allow_nan=False)
    with Path(path).open('x') as stream:
        stream.write(text)


def verify_bundle(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise RuntimeError(f'Bundle hash mismatch: {name}')
    for name, expected in manifest['external_dependencies'].items():
        if sha(name) != expected:
            raise RuntimeError(f'External dependency changed: {name}')
    return manifest


def snapshot(state, server_version):
    out = {k:list(getattr(state,k)) for k in ('q','dq','O_T_EE','F_T_EE','F_x_Cee','I_ee','tau_ext_hat_filtered','O_F_ext_hat_K')}
    out.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), read_only=True,
               server_version=server_version, robot_mode=str(state.robot_mode),
               current_errors=str(state.current_errors),last_motion_errors=str(state.last_motion_errors),
               m_ee=state.m_ee,m_load=state.m_load)
    return out


def state_checks(s, chain, reference):
    if s['server_version'] != 9:
        raise RuntimeError('Unexpected robot FCI version')
    for name in ('F_T_EE','F_x_Cee','I_ee','m_ee','m_load'):
        if not np.allclose(s[name],reference[name],rtol=0,atol=1e-7):
            raise RuntimeError(f'Controller tool/load configuration changed: {name}')
    actual=np.asarray(s['O_T_EE']).reshape(4,4,order='F')
    predicted=fk(chain,np.asarray(s['q']))
    error=np.linalg.norm(actual[:3,3]-predicted[:3,3])
    angle=np.arccos(np.clip((np.trace(actual[:3,:3].T@predicted[:3,:3])-1)/2,-1,1))
    if not np.isfinite(np.r_[s['q'],s['dq'],error,angle]).all() or error>.002 or angle>np.deg2rad(1):
        raise RuntimeError('Robot readback / URDF TCP mismatch')
    return dict(fixed_tcp_position_error_m=float(error),fixed_tcp_orientation_error_deg=float(np.rad2deg(angle)),
                idle=s['robot_mode']=='RobotMode.kIdle',at_rest=bool(max(abs(np.asarray(s['dq'])))<.01),
                current_errors=s['current_errors'],note='Internal configuration check; not independent physical calibration')


def require_motion_checks(s, checks, prepared, width):
    if not checks['idle'] or not checks['at_rest'] or s['current_errors'] != '[]':
        raise RuntimeError('Motion requires Idle, stationary, no current error. No automatic recovery.')
    first=evaluate(prepared['execution_plan'],0.)
    if max(abs(np.asarray(s['q'])-first[:7]))>.002 or abs(width-first[7])>.001:
        raise RuntimeError('State changed after preparation; generate and review a new start plan')


def require_stable_gripper(readings):
    widths = np.asarray([r['width_m'] for r in readings])
    if len(widths) < 20 or not np.isfinite(widths).all() or np.ptp(widths) > .0005:
        raise RuntimeError('Gripper must have fresh, stable width before trajectory preparation/execution')


def make_prepared(root, ep_number, phase, s, width, *, check_geometry=True):
    original=json.loads((root/'timed_plan.json').read_text())
    ep=next(e for e in original['episodes'] if e['episode']==ep_number)
    if phase in ('start-only', 'start-check'):
        target=evaluate(ep,0.)
        ep=(start_check(s['q'], width, target[:7]) if phase == 'start-check'
            else transit(s['q'],width,target[:7],target[7]))
    elif phase != 'replay':
        raise ValueError('Unknown execution phase')
    result=dict(episode=ep_number,phase=phase,execution_plan=ep,derivative_check=validate(ep),
                bundle_manifest_sha256=sha(root/'manifest.json'),source_plan_sha256=sha(root/'timed_plan.json'),
                snapshot=s,initial_measured_width_m=width,motion_commands_sent=0,
                hardware_released=False)
    if check_geometry:
        result['geometry']=geometry(ep,root/'model.urdf',json.loads((root/'scene.json').read_text()))
    return result


def execution_gate(root, prepared_path, prepared, confirmation, operator_release=None):
    # Explicit physical test release is separate from software deployment.
    # A geometry pass by itself is not a motion authorization.
    if confirmation != sha(prepared_path) or os.environ.get('FRANKA_PHYSICAL_WATCHER_CONFIRMED') != 'YES':
        raise RuntimeError('Missing exact plan-hash confirmation / physical observer confirmation')
    if prepared['bundle_manifest_sha256'] != sha(root/'manifest.json'):
        raise RuntimeError('Prepared plan belongs to another bundle version')
    report=prepared.get('geometry',{})
    if not report.get('sampled_geometry_pass'):
        raise RuntimeError('Geometry gate closed: unresolved collision report; no override switch')
    if operator_release is not None:
        from operator_replay import validate_release
        release=validate_release(root, operator_release, prepared, sha(prepared_path))
    else:
        release=json.loads((root/'physical_test_release.json').read_text())
    if not release.get('approved') or release.get('prepared_sha256') != sha(prepared_path):
        raise RuntimeError('No reviewed physical test release for this exact plan')
    age=time.time()-datetime.fromisoformat(prepared['snapshot']['timestamp_utc']).timestamp()
    if not 0 <= age <= 600:
        raise RuntimeError('Preparation snapshot expired (10 minutes); re-read and re-check')
    validate(prepared['execution_plan'])
    return release


def control(panda, core, ep, chain, box, result_dir):
    controller=core.TimedReplayCandidate(ep['times_s'],ep['coeff_descending_unit_interval'],chain['origins'],chain['axes'],chain['tail'])
    scheduler=GripperClock(ep)
    events=[]
    outcome=dict(completed=False,motion_started=False)
    seq,width=box.wait_fresh()
    controller.feed_gripper(seq,0.,width)
    # Never use enable_logging/get_state in the supervisor loop: native
    # telemetry is preallocated and the callback never waits for a disk writer.
    old_handlers={}
    def interrupted(signum,frame):
        controller.abort_external()
        raise KeyboardInterrupt(f'Signal {signum}')
    for sig in (signal.SIGINT,signal.SIGTERM):
        old_handlers[sig]=signal.signal(sig,interrupted)
    try:
        panda.start_controller_guarded(controller)
        outcome['motion_started']=True
        last_fresh=time.monotonic()
        while panda.control_thread_active():
            sample=box.fresh()
            if sample is not None:
                seq,width=sample
                phase=controller.get_time()
                controller.feed_gripper(seq,phase,width)
                last_fresh=time.monotonic()
                events.append([time.monotonic(),phase,'feedback',seq,width])
            if time.monotonic()-last_fresh > .2:
                raise RuntimeError('Supervisor gripper feedback timeout')
            if not panda.control_thread_active():
                break
            command=dispatch_gripper(controller,scheduler,box.send)
            if command is not None:
                events.append([time.monotonic(),controller.get_time(),'submitted',None,command])
            if len(events)>150000:
                raise RuntimeError('Bounded transport log full')
            time.sleep(.005)
        # Thread is done before reading its stored error or requesting a join.
        panda.stop_controller()
        panda.raise_error()
        if controller.fault_code() or not controller.completed():
            raise RuntimeError(f'Incomplete timed replay, fault={controller.fault_code()}')
        outcome['completed']=True
    except BaseException as exc:
        controller.abort_external()
        outcome['error']=f'{type(exc).__name__}: {exc}'
        # Only latch abort and let libfranka end control; no gripper reset/open
        # command, no error recovery, no automatic resume, no second trajectory.
        while panda.control_thread_active():
            time.sleep(.01)
        panda.stop_controller()
        try:
            panda.raise_error()
        except RuntimeError as native_error:
            outcome['native_error']=str(native_error)
    finally:
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
        outcome.update(fault_code=controller.fault_code(),arm_time_s=controller.get_time())
        np.savez_compressed(result_dir/'arm_telemetry.npz',data=controller.telemetry())
        save(result_dir/'transport_events.json',events)
        save(result_dir/'execution_result.json',outcome)
    return outcome


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['offline','read-only','prepare','execute'])
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=int,choices=[0,1],default=0)
    p.add_argument('--phase',choices=['start-check','start-only','replay'],default='start-only')
    p.add_argument('--prepared',type=Path)
    p.add_argument('--confirm-plan-sha256')
    p.add_argument('--operator-release',type=Path)
    a=p.parse_args()
    root=a.bundle.resolve()
    manifest=verify_bundle(root)
    os.chdir(root)
    config=json.loads((root/'config.json').read_text())
    plan=json.loads((root/'timed_plan.json').read_text())
    reference=json.loads((root/'tool_reference.json').read_text())
    chain=urdf_chain(root/'model.urdf')
    if chain != plan['chain']:
        raise RuntimeError('Bundled URDF differs from compiled FK chain')
    from panda_py import _core as core,libfranka
    require_candidate(core)
    if not hasattr(core.Panda,'control_thread_active') or not hasattr(core.TimedReplayCandidate,'telemetry'):
        raise RuntimeError('Wrong native staging build')
    prepared=None
    if a.mode=='execute':
        if a.prepared is None:raise RuntimeError('Explicit prepared file required')
        prepared=json.loads(a.prepared.read_text())
        release=execution_gate(root,a.prepared,prepared,a.confirm_plan_sha256,a.operator_release)
    a.output.mkdir(parents=True,exist_ok=False)
    # Lock is exclusive even for read-only SDK sessions; never steal UDP/socket
    # ownership from the recorder or another player.
    with (root/'session.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=dict(mode=a.mode,motion_commands_sent=0,hardware_released=False,
                    bundle_manifest_sha256=sha(root/'manifest.json'))
        try:
            if a.mode=='offline':
                result['episodes']=[validate(e) for e in plan['episodes']]
                for ep in plan['episodes']:
                    core.TimedReplayCandidate(ep['times_s'],ep['coeff_descending_unit_interval'],chain['origins'],chain['axes'],chain['tail'])
            else:
                if a.mode=='execute':
                    panda=core.Panda(config['robot_ip'],'p0_staged',libfranka.RealtimeConfig.kIgnore)
                    robot=panda.get_robot()
                else:
                    robot=libfranka.Robot(config['robot_ip'],libfranka.RealtimeConfig.kIgnore)
                s=snapshot(robot.read_once(),robot.server_version())
                result['robot']=s
                result['robot_checks']=state_checks(s,chain,reference)
                box=BoxTransport(config['box_sdk_dir'],config['gripper_device_id'],config['gripper_ip'],allow_commands=a.mode=='execute')
                try:
                    box.open()
                    readings=[]
                    for _ in range(20):
                        seq,width=box.wait_fresh()
                        readings.append(dict(host_monotonic_s=time.monotonic(),sequence=seq,width_m=width))
                    result['gripper']=dict(device_id=box.device_id,mode=box.mode,discovery=box.discovery,
                                           discovery_resolution=box.discovery_resolution,fresh_readings=readings)
                    if a.mode in ('prepare', 'execute'):
                        box.require_control_mode()
                        require_stable_gripper(readings)
                    if a.mode=='prepare':
                        prepared=make_prepared(root,a.episode,a.phase,s,width)
                        save(a.output/'prepared_plan.json',prepared)
                    elif a.mode=='execute':
                        # Re-read after BOX discovery, not from a stale Panda cache.
                        s=snapshot(robot.read_once(),robot.server_version())
                        require_motion_checks(s,state_checks(s,chain,reference),prepared,width)
                        # No mode change is permitted. Re-read both devices after
                        # the independent mode query and verify the exact start.
                        for _ in range(20):
                            seq,width=box.wait_fresh()
                        s=snapshot(robot.read_once(),robot.server_version())
                        result['execution_preflight'] = dict(robot=s, measured_width_m=width,
                                                             automatic_mode_switching=False)
                        require_motion_checks(s,state_checks(s,chain,reference),prepared,width)
                        result['motion_commands_sent']='execution attempted; consult native and transport logs'
                        result['execution']=control(panda,core,prepared['execution_plan'],chain,box,a.output)
                        result['motion_commands_sent']='see execution_result.json'
                        result['hardware_released']=True
                finally:
                    box.close()
        except BaseException as exc:
            result['error']=f'{type(exc).__name__}: {exc}'
        save(a.output/'report.json',result)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 1 if 'error' in result or result.get('execution',{}).get('completed') is False else 0


if __name__=='__main__':
    raise SystemExit(main())
