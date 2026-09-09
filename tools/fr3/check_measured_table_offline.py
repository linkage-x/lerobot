#!/usr/bin/env python3
"""Measured table wedge vs every collision mesh. Offline only; no SDK imports.

Broad-phase AABB distance is a lower bound, not an exact mesh distance.
Refine ambiguous (<10 mm) bounds with Coal mesh/box queries. The finite box
represents an infinite wedge only after proving the whole robot is inside
its far bounds. Mesh pairs within the robot are a separate check.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import coal
import numpy as np
import pinocchio as pin
from scipy.optimize import brentq

from replay_ik_trajectory_guarded import JOINTS, LOWER, UPPER, WALL_BAND, read_contact_ik


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distance_lower_bound(lo, hi, box_lo, box_hi):
    return np.linalg.norm(np.maximum(np.maximum(box_lo-hi, lo-box_hi), 0.), axis=-1)


def transit_values(q0, q1):
    delta = q1-q0
    magnitude = float(np.max(abs(delta)))
    # Analytic extrema of 10u^3-15u^4+6u^5, zero endpoint v/a.
    duration = 1.01*max(1., 1.875*magnitude/.12,
                       np.sqrt((10/np.sqrt(3))*magnitude/.4),
                       np.cbrt(60*magnitude/5.))
    count = int(np.ceil(max(duration/.020, 1.875*magnitude/.004)))
    u = np.linspace(0, 1, count+1)
    progress = 10*u**3-15*u**4+6*u**5
    q = q0+progress[:, None]*delta
    if np.min(np.minimum(q-LOWER, UPPER-q)-WALL_BAND) <= 0:
        raise ValueError('Proposed start transit enters a joint virtual-wall band')
    return u*duration, q


class MeshScene:
    def __init__(self, urdf, scene):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.geom = pin.buildGeomFromUrdf(self.model, str(urdf), pin.GeometryType.COLLISION,
                                          package_dirs=[str(Path(urdf).resolve().parent)])
        self.data = self.model.createData()
        self.gd = pin.GeometryData(self.geom)
        self.q = pin.neutral(self.model)
        self.arm_idx = [self.model.joints[self.model.getJointId(n)].idx_q for n in JOINTS]
        self.grip_idx = [self.model.joints[i].idx_q for i,n in enumerate(self.model.names)
                         if n.startswith('joint_gripper_')]
        if len(self.grip_idx) != 7:
            raise ValueError('Unexpected gripper joint mapping')
        self.max_angle = float(np.min(self.model.upperPositionLimit[self.grip_idx]))
        self.left = self.model.getFrameId('link_gripper_contact_left')
        self.right = self.model.getFrameId('link_gripper_contact_right')
        self.base_id = self.model.getFrameId('base')
        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.base = self.data.oMf[self.base_id].copy()
        corners = []
        self.mesh_hashes = {}
        for item in self.geom.geometryObjects:
            v = np.asarray(item.geometry.vertices())
            corners.append(list(itertools.product(*zip(v.min(0), v.max(0)))))
            self.mesh_hashes[item.name] = dict(path=item.meshPath, sha256=sha(item.meshPath))
        self.corners = np.asarray(corners)
        self.names = [o.name for o in self.geom.geometryObjects]
        self.minimum = np.full(len(corners), np.inf)
        self.threshold = scene['diagnostic_clearance_threshold_m']
        self.boxes = []
        for label, expansion in [('nominal',0.),('expanded_5mm',scene['extra_uncertainty_sensitivity_m'])]:
            lo = np.array([scene['table_front_x_m']-expansion,-3.,-3.])
            hi = np.array([3.,3.,scene['table_top_z_m']+expansion])
            self.boxes.append((label,lo,hi,coal.Box(*(hi-lo)),coal.Transform3s(np.eye(3),(lo+hi)/2)))
        self.width_cache = {}

    def opening(self, angle):
        self.q[self.grip_idx] = angle
        pin.framesForwardKinematics(self.model,self.data,self.q)
        return float(np.linalg.norm(self.data.oMf[self.left].translation-self.data.oMf[self.right].translation))

    def set_pose(self, arm, width):
        self.q[self.arm_idx] = arm
        key = float(width)
        if key not in self.width_cache:
            if key < 0 or key > self.opening(self.max_angle)+1e-8:
                raise ValueError('Gripper width outside URDF range')
            self.width_cache[key] = 0. if key < 1e-10 else brentq(lambda a:self.opening(a)-key,0.,self.max_angle,xtol=1e-12)
        self.q[self.grip_idx] = self.width_cache[key]
        pin.framesForwardKinematics(self.model,self.data,self.q)
        pin.updateGeometryPlacements(self.model,self.data,self.geom,self.gd,self.q)
        poses = [self.base.inverse()*p for p in self.gd.oMg]
        rotations = np.array([p.rotation for p in poses])
        positions = np.array([p.translation for p in poses])
        points = np.einsum('nij,nkj->nki',rotations,self.corners)+positions[:,None,:]
        lo,hi = points.min(1),points.max(1)
        # The box has a finite artificial far edge. Never rely on that edge for clearance.
        if np.max(hi[:,0]) >= 2.9 or np.max(abs(points[:,:,1])) >= 2.9 or np.min(lo[:,2]) <= -2.9:
            raise ValueError('Robot outside certified bounds of conservative wedge proxy')
        return poses,lo,hi

    def check(self, samples, label):
        nmesh=len(self.names)
        mins = {name:np.full(nmesh,np.inf) for name,*_ in self.boxes}
        worst = {name:[None]*nmesh for name,*_ in self.boxes}
        hits = {name:{} for name,*_ in self.boxes}
        queries=0
        count=0
        for count,(t,arm,width) in enumerate(samples,1):
            poses,lo,hi=self.set_pose(arm,width)
            for boxname,blo,bhi,box,boxpose in self.boxes:
                bounds=distance_lower_bound(lo,hi,blo,bhi)
                for i in np.flatnonzero(bounds < self.threshold):
                    request=coal.DistanceRequest()
                    result=coal.DistanceResult()
                    value=coal.distance(self.geom.geometryObjects[i].geometry,
                        coal.Transform3s(poses[i].rotation,poses[i].translation),box,boxpose,request,result)
                    if not np.isfinite(value):
                        raise ValueError('Nonfinite Coal distance')
                    bounds[i]=max(0.,float(value))
                    queries+=1
                    if value <= 1e-8:
                        entry=hits[boxname].setdefault(self.names[i],dict(samples=0,first_time_s=float(t)))
                        entry['samples']+=1;entry['last_time_s']=float(t)
                for i in np.flatnonzero(bounds < mins[boxname]):
                    mins[boxname][i]=bounds[i]
                    worst[boxname][i]=dict(time_s=float(t),q_rad=np.asarray(arm).tolist(),width_m=float(width))
            if count%2000 == 0:
                print(f'{label}: {count} poses checked',flush=True)
        cases={}
        for name,*_ in self.boxes:
            idx=int(np.argmin(mins[name]))
            cases[name]=dict(minimum_sampled_clearance_lower_bound_m=float(mins[name][idx]),
                limiting_mesh=self.names[idx],worst_sample=worst[name][idx],
                all_sampled_meshes_at_least_diagnostic_clearance=bool(np.all(mins[name]>=self.threshold)),
                intersection_meshes=hits[name],
                per_mesh_clearance_lower_bound_m=dict(zip(self.names,mins[name].tolist())))
        result=dict(label=label,samples=count,mesh_box_refinement_queries=queries,cases=cases)
        print(json.dumps(dict(label=label,samples=count,cases={k:{a:b for a,b in v.items() if a!='per_mesh_clearance_lower_bound_m'} for k,v in cases.items()})),flush=True)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['urdf','samples','scene','snapshot','output']:
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    scene=json.loads(a.scene.read_text());snapshot=json.loads(a.snapshot.read_text())
    if not snapshot.get('read_only') or len(snapshot['q']) != 7: raise ValueError('Invalid read-only start snapshot')
    episodes=read_contact_ik(a.samples)
    checker=MeshScene(a.urdf,scene)
    report=dict(hardware_ready=False,motion_commands_sent=0,continuous_collision_certified=False,
        scene=scene,script_sha256=sha(__file__),input_sha256={k:sha(getattr(a,k)) for k in ['urdf','samples','scene','snapshot']},
        mesh_hashes=checker.mesh_hashes,pinocchio_version=pin.__version__,coal_version=coal.__version__,
        snapshot_timestamp_utc=snapshot['timestamp_utc'],snapshot_robot_mode=snapshot['robot_mode'],
        base_pose_in_world=checker.base.homogeneous.tolist(),collision_mesh_count=len(checker.names),
        limitations=['Table front assumed parallel to base Y and top parallel to base XY.',
          'AABB broad-phase clearance is a lower bound, not an exact minimum mesh distance.',
          'Sampled geometry only; no continuous collision proof, stopping distance or physical tracking test.',
          'No self-collision queries in this table checker; previous replay self-check remains separate.',
          'Start paths are proposed joint-space quintic transits, not the existing native launcher path.',
          'Start paths check only three fixed gripper widths; current real width and intermediate opening sweep are not verified.',
          'No allowance for payload/objects above the table or obstacles on the robot side of its front edge.'],
        replay=[],start_candidates=[])
    for ep,seq in episodes.items():
        report['replay'].append(checker.check(((s['timestamp_s'],s['q'],s['width_m']) for s in seq),f'episode_{ep}'))
        times,joints=transit_values(np.asarray(snapshot['q']),seq[0]['q'])
        width_cases={'closed':0.,'half_open':.0443699024617672,'max_open':.0887398049235344}
        for name,width in width_cases.items():
            r=checker.check(((t,q,width) for t,q in zip(times,joints)),f'start_episode_{ep}_{name}')
            r.update(episode=ep,width_scenario=name,constant_width_m=width,duration_s=float(times[-1]),
                max_joint_step_rad=float(np.max(abs(np.diff(joints,axis=0)))),start_q_rad=snapshot['q'],end_q_rad=seq[0]['q'].tolist())
            report['start_candidates'].append(r)
    with a.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print('Offline table report saved; hardware replay remains unapproved.',flush=True)

if __name__=='__main__': main()
