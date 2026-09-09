"""Exercise staged native telemetry/completion on synthetic states only."""
import json
from pathlib import Path
import numpy as np
from panda_py import _core as core
from plans import transit
from replay_timed_candidate import GripperClock,dispatch_gripper,evaluate


def run(root):
    plan=json.loads((root/'timed_plan.json').read_text())
    chain=plan['chain'];e=plan['episodes'][0];first=evaluate(e,0.)
    ep=transit(first[:7],first[7],first[:7]+.002,first[7]-.001)
    c=core.TimedReplayCandidate(ep['times_s'],ep['coeff_descending_unit_interval'],chain['origins'],chain['axes'],chain['tail'])
    c.feed_gripper(1,0.,float(first[7]));c._offline_start(first[:7],np.zeros(7),c.fk(first[:7]))
    scheduler=GripperClock(ep)
    for tick in range(int((ep['duration_s']+.3)*1000)):
        t=tick*.001;c._offline_set_time(t);target=c.evaluate(t)
        if tick and tick%10==0:c.feed_gripper(tick+1,t,float(target[7]))
        if tick%5==0:dispatch_gripper(c,scheduler,lambda x:True)
        out=c._offline_step(t,target[:7],c.evaluate(t,1)[:7],c.fk(target[:7]),1)
        if out['finished']:break
    assert c.completed() and c.fault_code()==0
    rows=c.telemetry()
    assert rows.ndim==2 and rows.shape[1]==48 and len(rows)>200
    assert np.isfinite(rows).all() and np.min(np.diff(rows[:,0]))>=.005-1e-9
    np.testing.assert_allclose(rows[:,1:8],rows[:,31:38],atol=1e-12)
    for row in rows[::20]:
        np.testing.assert_allclose(row[15:31].reshape(4,4,order='F'),c.fk(row[1:8]),atol=1e-12)
    try:c._offline_start(first[:7],np.zeros(7),c.fk(first[:7]))
    except RuntimeError:pass
    else:raise AssertionError('Controller restarted')
    result=dict(physical_commands_sent=0,robot_instances_created=0,completed=True,telemetry_rows=len(rows),
                telemetry_columns=48,checks=['48-column telemetry shape','monotonic 5ms records','actual vs target joints',
                                          'TCP column-major layout','fresh-gripper completion','restart denied'])
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(run(Path(__file__).resolve().parent),f,indent=2)
