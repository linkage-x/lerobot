"""Exact executable polynomials and sampled geometry, never hardware access."""
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from replay_timed_candidate import bounds, evaluate, extrema, WIDTH_MAX
from replay_ik_trajectory_guarded import LOWER, UPPER, WALL_BAND


def validate(ep):
    t = np.asarray(ep['times_s'], dtype=float)
    c = np.asarray(ep['coeff_descending_unit_interval'], dtype=float)
    if len(t) < 2 or c.shape != (len(t)-1, 6, 8) or not np.isfinite(t).all() or not np.isfinite(c).all():
        raise ValueError('Invalid polynomial shape / finite values')
    if t[0] != 0 or np.min(np.diff(t)) <= 0 or t[-1] > 600 or t[-1] != ep['duration_s']:
        raise ValueError('Invalid trajectory clock')
    lo, hi, peak = bounds(t,c)
    if np.any(lo[:7] <= LOWER+WALL_BAND) or np.any(hi[:7] >= UPPER-WALL_BAND):
        raise ValueError('Trajectory enters virtual-wall bands')
    if lo[7] < -1e-12 or hi[7] > WIDTH_MAX+1e-12:
        raise ValueError('Opening out of range')
    if np.any(peak[:,:7] > np.array([.12,.4,5.])[:,None]+1e-7) or peak[0,7] > .0100001:
        raise ValueError('Derivative limits exceeded')
    for x in [0., t[-1]]:
        for d in [1,2]:
            if np.max(abs(evaluate(ep,x,d)[:7])) > 1e-7:
                raise ValueError('Arm start/end not at rest')
    return dict(duration_s=float(t[-1]), min_wall_clearance_rad=float(np.minimum(lo[:7]-LOWER-WALL_BAND,UPPER-WALL_BAND-hi[:7]).min()),
                maximum_derivatives=peak.tolist(), width_range_m=[float(lo[7]),float(hi[7])])


def transit(q0, width0, q1, width1):
    a, b = np.r_[q0,width0], np.r_[q1,width1]
    if a.shape != (8,) or b.shape != (8,) or not np.isfinite(np.r_[a,b]).all():
        raise ValueError('Invalid start state')
    delta = b-a
    m = max(abs(delta[:7]))
    duration = 1.01*max(1.,1.875*m/.12,np.sqrt(10/np.sqrt(3)*m/.4),np.cbrt(60*m/5.),1.875*abs(delta[7])/.01)
    c = np.zeros((1,6,8))
    c[0,0],c[0,1],c[0,2],c[0,5] = 6*delta,-15*delta,10*delta,a
    ep = dict(episode='start-only',duration_s=float(duration),times_s=[0.,float(duration)],coeff_descending_unit_interval=c.tolist())
    validate(ep)
    return ep


def samples(ep):
    t = np.asarray(ep['times_s'])
    c = np.asarray(ep['coeff_descending_unit_interval'])
    evaluator = dict(ep,times_s=t,coeff_descending_unit_interval=c)
    for i,dt in enumerate(np.diff(t)):
        speed = np.array([max(abs(x) for x in extrema(c[i,:,j],1,dt)) for j in range(8)])
        n = max(1,int(np.ceil(max(dt/.02,max(speed[:7])*dt/.004,speed[7]*dt/.00025))))
        for k in range(n+(i==len(c)-1)):
            phase = float(t[i]+dt*k/n)
            v = evaluate(evaluator,phase)
            yield phase,v[:7],float(v[7])


def start_check(q0, width, target):
    delta = np.asarray(target) - np.asarray(q0)
    fraction = min(1., .03 / max(float(np.max(abs(delta))), 1e-12))
    ep = transit(q0, width, np.asarray(q0) + fraction * delta, width)
    ep['episode'] = 'start-check'
    ep['duration_s'] = max(5., ep['duration_s'])
    ep['times_s'][-1] = ep['duration_s']
    validate(ep)
    return ep


def geometry(ep, urdf, scene):
    from check_measured_table_offline import MeshScene
    import pinocchio as pin
    checker = MeshScene(urdf,scene)
    from hinge_contacts import ReviewedHinges
    reviewed = ReviewedHinges(checker, urdf)
    adjacent = {frozenset((j.find('parent').get('link'),j.find('child').get('link'))) for j in ET.parse(urdf).findall('joint')}
    exclusions = []
    for i,j in itertools.combinations(range(len(checker.names)),2):
        a,b = checker.geom.geometryObjects[i],checker.geom.geometryObjects[j]
        links = frozenset((checker.model.frames[a.parentFrame].name,checker.model.frames[b.parentFrame].name))
        if a.parentJoint == b.parentJoint or links in adjacent:
            exclusions.append([a.name,b.name])
        else:
            checker.geom.addCollisionPair(pin.CollisionPair(i,j))
    checker.gd = pin.GeometryData(checker.geom)
    hits = {}
    allowed = {}
    def audit_samples():
        for phase,q,width in samples(ep):
            checker.set_pose(q,width)
            pin.computeCollisions(checker.geom,checker.gd,False)
            for pair,result in zip(checker.geom.collisionPairs,checker.gd.collisionResults):
                if result.isCollision():
                    key=checker.names[pair.first]+' / '+checker.names[pair.second]
                    accepted, detail = reviewed.classify(pair.first, pair.second)
                    if accepted:
                        h = allowed.setdefault(key, dict(count=0, max_radial_m=0., max_axial_abs_m=0., max_contacts=0))
                        h['count'] += 1
                        h['max_radial_m'] = max(h['max_radial_m'], detail['radial_max_m'])
                        h['max_axial_abs_m'] = max(h['max_axial_abs_m'], detail['axial_abs_max_m'])
                        h['max_contacts'] = max(h['max_contacts'], detail['contacts'])
                        continue
                    h=hits.setdefault(key,dict(count=0,first_time_s=phase))
                    h['count']+=1;h['last_time_s']=phase
                    h['last_detail'] = detail
            yield phase,q,width
    table=checker.check(audit_samples(),str(ep['episode']))
    return dict(table=table,self_intersections=hits,excluded_rigid_or_direct_parent_pairs=exclusions,
                reviewed_local_pin_contacts=allowed,hinge_review=reviewed.description(),
                sampled_geometry_pass=not hits and all(x['all_sampled_meshes_at_least_diagnostic_clearance'] for x in table['cases'].values()),
                continuous_collision_certified=False,mesh_hashes=checker.mesh_hashes,
                note='No payload, cables, second arm or unspecified obstacles; exclusions require assembly review. 10mm is diagnostic clearance, not stopping distance.')
