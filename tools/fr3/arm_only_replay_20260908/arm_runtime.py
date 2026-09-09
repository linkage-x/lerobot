"""Independent arm-only slow replay; no BOX SDK, fake feedback or gripper commands.

The passive, empty gripper must physically remain at the declared width.
This tests arm motion, not width-dependent contact-point task reproduction.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import uuid
import numpy as np
from plans import validate,geometry,transit,start_check
from replay_timed_candidate import evaluate,fk,urdf_chain,WIDTH_MAX
from replay_ik_trajectory_guarded import require_candidate

ROOT=Path(__file__).resolve().parent
SOURCE_SHA='7fa69ec0c215de460095a0707d01fc93c850de5474b45d70296db21b6aa4a335'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,obj):
    text=json.dumps(obj,indent=2,allow_nan=False)
    with path.open('x') as f:f.write(text)


def verify(root):
    m=json.loads((root/'manifest.json').read_text())
    for path,digest in m['files'].items():
        p=(root/path).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:
            raise RuntimeError(f'Changed arm-only dependency: {path}')
    if sha('/usr/local/lib/libfranka.so.0.15.0')!=m['libfranka_sha256']:
        raise RuntimeError('Changed libfranka')
    if sha(root/'timed_plan.json')!=SOURCE_SHA:
        raise RuntimeError('Unexpected old IK / calibration source')
    if not json.loads((root/'offline_validation.json').read_text())['passed']:
        raise RuntimeError('Arm-only offline tests not passed')
    return m


def passive_plan(ep,width):
    if not np.isfinite(width) or not 0<=width<=WIDTH_MAX:
        raise ValueError('Width must be a measured value in 0..88.739804924 mm')
    result=copy.deepcopy(ep)
    c=np.asarray(result['coeff_descending_unit_interval'],float)
    before=c[:,:,:7].copy()
    c[:,:,7]=0.;c[:,5,7]=width
    result['coeff_descending_unit_interval']=c.tolist()
    result['passive_gripper_declared_width_m']=width
    result['gripper_measured']=False
    assert np.array_equal(c[:,:,:7],before)
    validate(result)
    return result


def snapshot(robot):
    s=robot.read_once()
    return dict(q=list(s.q),dq=list(s.dq),O_T_EE=list(s.O_T_EE),F_T_EE=list(s.F_T_EE),
                F_x_Cee=list(s.F_x_Cee),I_ee=list(s.I_ee),m_ee=s.m_ee,m_load=s.m_load,
                robot_mode=str(s.robot_mode),current_errors=str(s.current_errors),
                last_motion_errors=str(s.last_motion_errors),server_version=robot.server_version(),
                timestamp_utc=datetime.now(timezone.utc).isoformat())


def state_checks(s,chain,reference,expected=None):
    if s['server_version']!=9 or s['robot_mode']!='RobotMode.kIdle' or s['current_errors']!='[]':
        raise RuntimeError('Robot must be Idle/error-free, protocol 9; no automatic recovery')
    for name in ('F_T_EE','F_x_Cee','I_ee','m_ee','m_load'):
        if not np.allclose(s[name],reference[name],atol=1e-7,rtol=0):
            raise RuntimeError(f'Tool configuration changed: {name}')
    q=np.asarray(s['q']);dq=np.asarray(s['dq'])
    actual=np.asarray(s['O_T_EE']).reshape(4,4,order='F');predicted=fk(chain,q)
    pos=float(np.linalg.norm(actual[:3,3]-predicted[:3,3]))
    angle=float(np.arccos(np.clip((np.trace(actual[:3,:3].T@predicted[:3,:3])-1)/2,-1,1)))
    if not np.isfinite(np.r_[q,dq,pos,angle]).all() or abs(dq).max()>.01 or pos>.002 or angle>np.deg2rad(1):
        raise RuntimeError('Robot moving or controller/URDF mismatch')
    if expected is not None and abs(q-np.asarray(expected)).max()>.002:
        raise RuntimeError('Robot moved after preparation; no unchecked start jump')
    return dict(tcp_position_error_m=pos,tcp_angle_error_deg=float(np.rad2deg(angle)))


def make_controller(core,ep,chain):
    return core.ArmOnlyReplay(ep['times_s'],ep['coeff_descending_unit_interval'],
                             chain['origins'],chain['axes'],chain['tail'])


def control(panda,core,ep,chain,out):
    out.mkdir()
    c=make_controller(core,ep,chain)
    result=dict(completed=False,motion_started=False,gripper_commands_sent=0,gripper_sdk_loaded=False,
                declared_width_m=ep['passive_gripper_declared_width_m'],gripper_width_measured=False)
    handlers={}
    def interrupt(sig,frame):
        c.abort_external()
        raise KeyboardInterrupt(f'Signal {sig}')
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
    try:
        panda.start_controller_guarded(c)
        result['motion_started']=True
        while panda.control_thread_active():time.sleep(.01)
        panda.stop_controller();panda.raise_error()
        if c.fault_code() or not c.completed():raise RuntimeError(f'Arm-only fault {c.fault_code()}')
        result['completed']=True
    except BaseException as exc:
        c.abort_external()
        result['error']=f'{type(exc).__name__}: {exc}'
        while panda.control_thread_active():time.sleep(.01)
        panda.stop_controller()
        try:panda.raise_error()
        except RuntimeError as err:result['native_error']=str(err)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        result.update(arm_time_s=c.get_time(),fault_code=c.fault_code())
        np.savez_compressed(out/'arm_telemetry.npz',data=c.telemetry())
        save(out/'execution_result.json',result)
    if not result['completed']:raise RuntimeError(f'Arm-only stage stopped; no retry: {out}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=int,choices=[0,1])
    p.add_argument('--gripper-width-mm',type=float)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--start-only',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.execute and a.prepare_only:p.error('Select execute OR prepare-only')
    verify(ROOT)
    if a.gripper_width_mm is None:
        if not sys.stdin.isatty():p.error('Interactive measured gripper width is required')
        print('只测试机械臂；不连接夹爪、不自动开合。请先取下夹持物并清空运动范围。\n'
              '夹爪必须能保持开口不变；若会自行开合，请取消。不要把“断网”当作机械锁定。',flush=True)
        a.gripper_width_mm=float(input('输入两指尖当前实际净距离（mm，不知道则 Ctrl+C 取消）：'))
    width=a.gripper_width_mm/1000
    if a.execute:
        if not sys.stdin.isatty():p.error('Real motion requires a local operator terminal')
        if input('确认夹爪空夹且会保持该开口、现场清空、急停可用；输入 YES 开始只动臂流程：').strip()!='YES':
            print('已取消，未发运动命令。');return 2
    from panda_py import _core as core,libfranka
    require_candidate(core)
    if getattr(core,'_FR3_ARM_ONLY_API',None)!=1:raise RuntimeError('Wrong arm-only build')
    chain=urdf_chain(ROOT/'model.urdf')
    source=json.loads((ROOT/'timed_plan.json').read_text())
    if chain!=source['chain']:raise RuntimeError('Source/model chain mismatch')
    ep=passive_plan(next(x for x in source['episodes'] if x['episode']==a.episode),width)
    reference=json.loads((ROOT/'tool_reference.json').read_text())
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    out=logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir()
    print(f'日志目录：{out}',flush=True)
    summary=dict(episode=a.episode,gripper_commands_sent=0,gripper_sdk_loaded=False,
                 gripper_width_measured=False,declared_width_m=width,phases=[],motion_started=False)
    # Also take the old player's device lock, so the two versions cannot race.
    with (ROOT/'session.lock').open('a') as lock, Path('/home/nvidia/box_api/replay_p0_once_20260908/session.lock').open('a') as oldlock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            robot=libfranka.Robot('192.168.11.102',libfranka.RealtimeConfig.kIgnore)
            s=snapshot(robot);state_checks(s,chain,reference);del robot;gc.collect()
            first=evaluate(ep,0.)[:7]
            check=passive_plan(start_check(s['q'],width,first),width)
            check_end=evaluate(check,check['duration_s'])[:7]
            to_first=passive_plan(transit(check_end,width,first,width),width)
            phases=[('start-check',check),('start-only',to_first)]+([] if a.start_only else [('replay',ep)])
            # Every phase is audited before any motion. No unchecked replanning
            # or skipping a failed stage; exact polynomials are executed later.
            for name,plan in phases:
                print(f'只读检查 {name}，预计运动 {plan["duration_s"]:.1f} s',flush=True)
                report=geometry(plan,ROOT/'model.urdf',json.loads((ROOT/'scene.json').read_text()))
                save(out/f'{name}_prepared.json',dict(plan=plan,geometry=report,snapshot=s))
                if not report['sampled_geometry_pass']:raise RuntimeError(f'{name}: passive-gripper geometry gate failed')
            if not a.execute:
                summary['prepared_only']=True
                print('只读准备完成，未发送运动命令。',flush=True)
                return 0
            # Recheck immediately before movement: preparation may take minutes.
            panda=core.Panda('192.168.11.102','p0_arm_only',libfranka.RealtimeConfig.kIgnore)
            for name,plan in phases:
                state_checks(snapshot(panda.get_robot()),chain,reference,evaluate(plan,0.)[:7])
                print(f'开始只动臂 {name}；夹爪无控制/无反馈。',flush=True)
                summary['motion_started']=True
                summary['phases'].append(dict(phase=name,result=control(panda,core,plan,chain,out/name)))
            summary['completed']=True
            print('本条机械臂轨迹完成。夹爪未连接、未发送开合命令。',flush=True)
        except BaseException as exc:
            summary['error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:save(out/'summary.json',summary)
    return 0


if __name__=='__main__':raise SystemExit(main())
