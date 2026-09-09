#!/usr/bin/env python3
"""Separate sampled self-collision audit of proposed start paths. No hardware."""
import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
from check_measured_table_offline import MeshScene, sha, transit_values
from replay_ik_trajectory_guarded import read_contact_ik

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['urdf','samples','scene','snapshot','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=MeshScene(a.urdf,json.loads(a.scene.read_text()))
    adjacent={frozenset((j.find('parent').get('link'),j.find('child').get('link'))) for j in ET.parse(a.urdf).findall('joint')}
    excluded=[]
    for i,j in itertools.combinations(range(len(c.geom.geometryObjects)),2):
        x,y=c.geom.geometryObjects[i],c.geom.geometryObjects[j]
        links=frozenset((c.model.frames[x.parentFrame].name,c.model.frames[y.parentFrame].name))
        reason='same rigid body' if x.parentJoint==y.parentJoint else 'direct parent-child; not a mechanically certified whitelist' if links in adjacent else None
        if reason:excluded.append(dict(pair=[x.name,y.name],reason=reason))
        else:c.geom.addCollisionPair(pin.CollisionPair(i,j))
    c.gd=pin.GeometryData(c.geom)
    snapshot=json.loads(a.snapshot.read_text())
    episodes=read_contact_ik(a.samples)
    report=dict(hardware_ready=False,motion_commands_sent=0,continuous_collision_certified=False,
        script_sha256=sha(__file__),inputs={n:sha(getattr(a,n)) for n in ['urdf','samples','scene','snapshot']},
        collision_pairs=len(c.geom.collisionPairs),excluded_pairs=excluded,
        note='Original collision meshes retained; no hinge overlaps suppressed. Three fixed-width cases only, not a full width sweep.',scenarios=[])
    for ep,seq in episodes.items():
        times,joints=transit_values(np.asarray(snapshot['q']),seq[0]['q'])
        for name,width in [('closed',0.),('half_open',.0443699024617672),('max_open',.0887398049235344)]:
            hits={}
            for i,(t,q) in enumerate(zip(times,joints)):
                c.set_pose(q,width)
                pin.computeCollisions(c.geom,c.gd,False)
                for pair,result in zip(c.geom.collisionPairs,c.gd.collisionResults):
                    if result.isCollision():
                        label=c.names[pair.first]+' / '+c.names[pair.second]
                        entry=hits.setdefault(label,dict(samples=0,first_time_s=float(t)))
                        entry['samples']+=1;entry['last_time_s']=float(t)
                if (i+1)%500==0:print(f'start episode {ep} / {name}: {i+1}/{len(times)}',flush=True)
            result=dict(episode=ep,width_scenario=name,width_m=width,samples=len(times),duration_s=float(times[-1]),intersecting_pairs=hits)
            report['scenarios'].append(result);print(json.dumps(result),flush=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
