"""Prepare and audit the latest two episodes using the reused native planner.

Offline ONLY. No robot/BOX objects or movement entry are provided. The native
planner does not constrain jerk; a valid planner result is not hardware release.
"""
import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from benchmark_native import SOURCE_SHA, digest, knots, chunk_indices

ROOT = Path(__file__).resolve().parent
VELOCITY = np.array([2.175]*4+[2.610]*3)*.01
ACCELERATION = np.array([15,7.5,10,12.5,15,20,20])*.01
MODEL_SHA = '2e10116497fac7ea8fe4f4217d9622a62f424ec32c1eb5aed088060e43e78535'
MANIFEST_SHA = 'bde90e537fb5b252f4762f531da9e97a31db76f32d57f9afa91a98974b113786'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def interpolation_index(s, anchors):
    if not np.isfinite(anchors).all() or np.min(np.diff(anchors)) <= 0:
        raise ValueError('Native source-frame anchors are not strictly increasing')
    if np.min(s) < -1e-8 or np.max(s) > anchors[-1]+1e-8:
        raise ValueError('Native progress outside source frame range')
    return np.interp(s, anchors, np.arange(len(anchors)))


def audit(ep, core, bundle, output, deviation):
    src = knots(ep)
    offsets = np.asarray(ep['times_s'])
    from replay_timed_candidate import evaluate
    from replay_ik_trajectory_guarded import LOWER, UPPER, WALL_BAND
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(bundle/'model.urdf'))
    pdata = model.createData()
    indexes = [model.joints[model.getJointId(f'fr3_joint{i}')].idx_q for i in range(1,8)]
    base = model.getFrameId('base')
    tcp = model.getFrameId('corenetic_gripper_ee')
    config = pin.neutral(model)

    def contact(q, width):
        config[indexes] = q
        pin.framesForwardKinematics(model, pdata, config)
        pose = (pdata.oMf[base].inverse()*pdata.oMf[tcp]).homogeneous.copy()
        d = width*1000
        radicand = 49.699345**2 - 5.474953*d - d*d/4
        if radicand < -1e-6:
            raise ValueError('Width outside contact TCP definition')
        z = (49.699345-np.sqrt(max(0.,radicand)))/1000
        pose[:3,3] -= pose[:3,2]*z
        return pose

    arrays = []
    chunk_reports = []
    elapsed = 0.
    for num, (a,b) in enumerate(chunk_indices(len(src),200)):
        path = core.Path(src[a:b,:7].tolist(), deviation)
        native = core.Trajectory(path, VELOCITY, ACCELERATION, .001)
        if not native.valid() or not 0 < native.duration() < 600:
            raise RuntimeError('Native planner failed')
        t = np.linspace(0.,native.duration(),int(np.ceil(native.duration()/.001))+1)
        state = np.array([native.state(float(x)) for x in t])
        if not np.isfinite(state).all():
            raise RuntimeError('Nonfinite native state')
        anchors = np.asarray(native.waypoint_path_positions())
        assert len(anchors) == b-a
        progress = interpolation_index(state[:,21], anchors)+a
        width_spline = PchipInterpolator(anchors,src[a:b,7])
        s = np.clip(state[:,21], anchors[0],anchors[-1])
        width = width_spline(s)
        width_speed = width_spline(s,1)*state[:,22]
        derivative_jump = np.diff(state[:,14:21],axis=0)/np.diff(t)[:,None]
        maxpos, maxangle, maxq, maxwidth = 0.,0.,0.,0.
        # Compare corresponding source progress, not nearest point on another
        # part of a looping path. Includes the width-dependent contact center.
        for i in range(0,len(t),20):
            source_time = np.interp(progress[i],np.arange(len(offsets)),offsets)
            source_value = evaluate(ep,float(source_time))
            actual = contact(state[i,:7],width[i])
            reference = contact(source_value[:7],source_value[7])
            maxpos = max(maxpos,float(np.linalg.norm(actual[:3,3]-reference[:3,3])))
            maxangle = max(maxangle,float(np.rad2deg(np.arccos(np.clip((np.trace(actual[:3,:3].T@reference[:3,:3])-1)/2,-1,1)))))
            maxq = max(maxq,float(abs(state[i,:7]-source_value[:7]).max()))
            maxwidth = max(maxwidth,float(abs(width[i]-source_value[7])))
        chunk_reports.append(dict(source_start=a,source_end=b,duration_s=float(t[-1]),
            max_velocity_rad_s=float(abs(state[:,7:14]).max()),
            max_acceleration_rad_s2=float(abs(state[:,14:21]).max()),
            sampled_acceleration_change_per_s=float(abs(derivative_jump).max()),
            max_gripper_speed_m_s=float(abs(width_speed).max()),
            min_virtual_wall_clearance_rad=float(np.minimum(state[:,:7]-LOWER-WALL_BAND,UPPER-WALL_BAND-state[:,:7]).min()),
            endpoint_error_rad=float(max(abs(state[0,:7]-src[a,:7]).max(),abs(state[-1,:7]-src[b-1,:7]).max())),
            comparison_20ms=dict(max_contact_position_difference_m=maxpos,max_orientation_difference_deg=maxangle,
                                max_joint_difference_rad=maxq,max_width_difference_m=maxwidth)))
        # No wall-clock fraction for the gripper: actual native arc progress.
        arrays.append(np.c_[t+elapsed,state[:,:21],width,width_speed,progress])
        elapsed += t[-1]
    values = np.concatenate([x if i == 0 else x[1:] for i,x in enumerate(arrays)])
    artifact = output/f'episode_{ep["episode"]}_native_samples.npz'
    np.savez_compressed(artifact, data=values)
    report = dict(episode=ep['episode'],source_frames=len(src),native_chunks=chunk_reports,
        native_duration_s=float(elapsed),previous_duration_s=ep['duration_s'],
        original_recording_duration_s=float(ep['original_times_s'][-1]),
        native_speed_factor=.01,max_deviation_rad=deviation,
        gripper_sync='PCHIP(width vs native arc coordinate); not elapsed-time percentage',
        sampled_artifact_sha256=digest(artifact),motion_commands_sent=0,hardware_released=False,
        release_blockers=['Native planner is second-order; no finite jerk guarantee or C2 acceleration continuity.',
                          'Gripper watchdog failure during real movement has not been reproduced/resolved.'])
    limits_ok = all(c['max_velocity_rad_s']<=.1200001 and c['max_acceleration_rad_s2']<=.4000001
                    and c['sampled_acceleration_change_per_s']<=5.000001
                    and c['max_gripper_speed_m_s']<=.0100001 and c['min_virtual_wall_clearance_rad']>0
                    for c in chunk_reports)
    report['sampled_existing_motion_envelope_pass'] = limits_ok
    if not limits_ok:
        report['release_blockers'].append('Sampled native output exceeds the current motion envelope; limits were NOT relaxed.')
    if deviation > 0:
        report['release_blockers'].append('Rounded-corner diagnostic only; this modifies the source path and is not exact replay.')
    # Reuse existing full-link/table checker as a pure function, with samples
    # from this native artifact. Restore its sampler immediately afterwards.
    import plans
    original_sampler = plans.samples
    selected = np.unique(np.r_[np.arange(0,len(values),20),len(values)-1])
    def sampler(_):
        for i in selected:
            row = values[i]
            yield float(row[0]),row[1:8],float(row[22])
    try:
        plans.samples = sampler
        report['geometry'] = plans.geometry({'episode':ep['episode']},bundle/'model.urdf',json.loads((bundle/'scene.json').read_text()))
    finally:
        plans.samples = original_sampler
    write(output/f'episode_{ep["episode"]}_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='geometry'},indent=2),flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=int, choices=[0,1])
    p.add_argument('--bundle',type=Path,default=Path('/home/nvidia/box_api/replay_p0_once_20260908'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic-rounding',type=float,choices=[0.,.0001,.02],default=0.)
    args = p.parse_args()
    if digest(args.bundle/'manifest.json') != MANIFEST_SHA or digest(args.bundle/'model.urdf') != MODEL_SHA or digest(args.bundle/'timed_plan.json') != SOURCE_SHA:
        raise RuntimeError('Source bundle changed: re-audit provenance; do not silently load another IK/model')
    manifest = json.loads((args.bundle/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(args.bundle/name) != expected:
            raise RuntimeError(f'Source dependency changed: {name}')
    sys.path.insert(0,str(args.bundle))
    sys.path.insert(0,str(ROOT/'build'))
    import _native_plan_audit as core
    args.output.mkdir(parents=True,exist_ok=False)
    logging.getLogger().setLevel(logging.CRITICAL)
    plan = json.loads((args.bundle/'timed_plan.json').read_text())
    ep = next(e for e in plan['episodes'] if e['episode']==args.episode)
    provenance = dict(source_plan_sha256=SOURCE_SHA, source_model_sha256=MODEL_SHA,
        source_manifest_sha256=MANIFEST_SHA,planner_extension_sha256=digest(core.__file__),
        new_sources={str(f.relative_to(ROOT)):digest(f) for f in ROOT.rglob('*') if f.is_file()
                     and ('native' in f.relative_to(ROOT).parts or f.parent==ROOT) and f.suffix in ('.cpp','.h','.py','.sh','.txt')},
        changes='Motion path construction/planning unchanged; corrected arc anchor bookkeeping and added exact derivative audit.',
        hardware_released=False,motion_commands_sent=0)
    write(args.output/'provenance.json',provenance)
    audit(ep,core,args.bundle,args.output,args.diagnostic_rounding)
    # Nonzero exit prevents an audit-only artifact being mistaken for release.
    return 3


if __name__=='__main__':
    raise SystemExit(main())
