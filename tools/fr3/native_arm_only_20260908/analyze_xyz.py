"""Read saved native plans/telemetry only; no device access."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation


def fk(chain,q):
    pose=np.eye(4)
    for o,a,x in zip(chain['origins'],chain['axes'],q):
        r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(np.asarray(a)*x).as_matrix()
        pose=pose@o@r
    return pose@chain['tail']


def stats(e):
    return dict(samples=len(e),signed_mean_mm=e.mean(axis=0).tolist(),
                mae_mm=abs(e).mean(axis=0).tolist(),rmse_mm=np.sqrt((e*e).mean(axis=0)).tolist(),
                p95_abs_mm=np.percentile(abs(e),95,axis=0).tolist(),
                max_abs_mm=abs(e).max(axis=0).tolist(),
                spatial_rmse_mm=float(np.sqrt((e*e).sum(axis=1).mean())),
                max_spatial_mm=float(np.linalg.norm(e,axis=1).max()),
                last_error_mm=e[-1].tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('source',type=Path)
    a=p.parse_args();chain=json.loads(a.source.read_text())['chain']
    result=dict(error_definition='actual minus same-time native planned TCP, millimetres',
                frame='Franka base; not camera/world axes',
                source='controller O_T_EE readback; not independent physical measurement',stages={})
    full=[];full_contact=[];plots=[]
    for name in ['start','replay_0','replay_1','replay_2']:
        d=a.log/name
        if not (d/'telemetry.npz').exists():continue
        telemetry=np.load(d/'telemetry.npz')['data']
        plan=np.load(d/'native_plan.npz');t=telemetry[:,0]
        assert telemetry.shape[1]==39 and np.isfinite(telemetry).all()
        q=np.array([np.interp(t,plan['t'],plan['data'][:,j]) for j in range(7)]).T
        target=np.array([fk(chain,x) for x in q])
        measured=np.array([x[15:31].reshape(4,4,order='F') for x in telemetry])
        actual_fk=np.array([fk(chain,x) for x in telemetry[:,1:8]])
        e=(measured[:,:3,3]-target[:,:3,3])*1000
        model_e=(actual_fk[:,:3,3]-target[:,:3,3])*1000
        audit=json.loads((d/'audit.json').read_text());width=audit['declared_width_m']*1000
        z=(49.699345-np.sqrt(49.699345**2-5.474953*width-width**2/4))/1000
        contact_e=((measured[:,:3,3]-measured[:,:3,2]*z)-(target[:,:3,3]-target[:,:3,2]*z))*1000
        orientation=Rotation.from_matrix(np.transpose(target[:,:3,:3],(0,2,1))@measured[:,:3,:3]).magnitude()*180/np.pi
        entry=dict(fixed_tcp=stats(e),same_model_fk_tracking=stats(model_e),
                   estimated_passive_contact_tcp=stats(contact_e),
                   controller_vs_urdf_max_mm=float(np.linalg.norm(measured[:,:3,3]-actual_fk[:,:3,3],axis=1).max()*1000),
                   orientation_rms_deg=float(np.sqrt((orientation**2).mean())),
                   measured_time_span_s=[float(t[0]),float(t[-1])],planned_duration_s=float(plan['t'][-1]),
                   max_sampling_gap_s=float(np.diff(t).max()),min_command_success_rate=float(telemetry[:,31].min()),
                   execution=json.loads((d/'execution.json').read_text()))
        result['stages'][name]=entry
        if name.startswith('replay'):
            full.append(e);full_contact.append(contact_e);plots.append((name,t,e))
    result['executed_replay_only']=stats(np.concatenate(full))
    result['executed_replay_contact_estimate']=stats(np.concatenate(full_contact))
    result['full_episode_completed']=all((a.log/f'replay_{i}'/'execution.json').exists() and
        json.loads((a.log/f'replay_{i}'/'execution.json').read_text())['completed'] for i in range(3))
    (a.log/'xyz_error_report.json').write_text(json.dumps(result,indent=2))
    try:
        import matplotlib
    except ImportError:
        print(json.dumps(result,indent=2))
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(11,7),sharex=True)
    offset=0
    for name,t,e in plots:
        for j,ax in enumerate(axs):
            ax.plot(t+offset,e[:,j],lw=.9,label=name)
            ax.axhline(0,color='black',lw=.5);ax.set_ylabel('XYZ'[j]+' error (mm)');ax.grid(alpha=.25)
        offset+=t[-1]
    axs[0].legend();axs[-1].set_xlabel('Executed replay time (s); inspection pauses excluded')
    fig.suptitle('Actual - native planned fixed TCP in Franka base\nPartial Episode 0; start approach excluded')
    fig.tight_layout();fig.savefig(a.log/'xyz_error.png',dpi=160)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
