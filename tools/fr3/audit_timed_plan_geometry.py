#!/usr/bin/env python3
"""Export bounded-step samples of the NEW interpolant for FCL geometry checks.

This is not continuous collision detection. TCP deviation is against the
joint-knot FK/linear-contact/Slerp reference, not independent physical truth.
"""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from replay_ik_trajectory_guarded import JOINTS, sha256
from replay_timed_candidate import evaluate, extrema, fk, transform


def retreat(width):
    d = width*1000
    return (49.699345-np.sqrt(49.699345**2-5.474953*d-d*d/4))/1000


def export(plan, urdf, csv_path):
    chain = plan["chain"]
    world_base = transform(ET.parse(urdf).find("./joint[@name='world_to_base']/origin"))
    stats = []
    fields = ["episode_index", "frame_index", "timestamp_s", "gripper_width_m", "tcp_retreat_m", "ik_ok"] + JOINTS + [f"contact_target_{a}_m" for a in "xyz"]
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for ep0 in plan["episodes"]:
            ep = dict(ep0, times_s=np.asarray(ep0["times_s"]), coeff_descending_unit_interval=np.asarray(ep0["coeff_descending_unit_interval"]))
            samples, max_xyz, max_angle, max_q_delta = 0, 0., 0., 0.
            max_dt, max_q_step, max_width_step = 0., 0., 0.
            previous = None
            for i, dt in enumerate(np.diff(ep["times_s"])):
                c = ep["coeff_descending_unit_interval"][i]
                speed = np.array([max(abs(v) for v in extrema(c[:,j],1,dt)) for j in range(8)])
                count = int(np.ceil(max(dt/.020, (speed[:7]*dt/.004).max(), speed[7]*dt/.00025)))
                start, end = ep["times_s"][i:i+2]
                a, b = evaluate(ep,start), evaluate(ep,end)
                fa, fb = fk(chain,a[:7]), fk(chain,b[:7])
                ca = fa[:3,3] - fa[:3,2]*retreat(a[7])
                cb = fb[:3,3] - fb[:3,2]*retreat(b[7])
                slerp = Slerp([0,1], Rotation.from_matrix([fa[:3,:3],fb[:3,:3]]))
                for n in range(count + (1 if i == len(ep["times_s"])-2 else 0)):
                    u = n/count
                    t = start + dt*u
                    value = evaluate(ep,t)
                    fixed = fk(chain,value[:7])
                    center = fixed[:3,3] - fixed[:3,2]*retreat(value[7])
                    world_center = world_base[:3,:3] @ center + world_base[:3,3]
                    max_xyz = max(max_xyz, np.linalg.norm(center-((1-u)*ca+u*cb)))
                    max_angle = max(max_angle, (slerp(u).inv()*Rotation.from_matrix(fixed[:3,:3])).magnitude())
                    max_q_delta = max(max_q_delta, np.max(abs(value[:7]-((1-u)*a[:7]+u*b[:7]))))
                    if previous is not None:
                        max_dt = max(max_dt,t-previous[0])
                        max_q_step = max(max_q_step,np.max(abs(value[:7]-previous[1][:7])))
                        max_width_step = max(max_width_step,abs(value[7]-previous[1][7]))
                    previous = (t,value)
                    row = dict(episode_index=ep["episode"],frame_index=samples,timestamp_s=float(t),
                               gripper_width_m=float(value[7]),tcp_retreat_m=float(retreat(value[7])),ik_ok=True)
                    row.update(zip(JOINTS,value[:7],strict=True))
                    row.update(zip([f"contact_target_{a}_m" for a in "xyz"],world_center,strict=True))
                    writer.writerow(row)
                    samples += 1
            result = dict(episode=ep["episode"],samples=samples,maximum_time_step_s=float(max_dt),
                          maximum_joint_step_rad=float(max_q_step),maximum_width_step_m=float(max_width_step),
                          maximum_contact_deviation_from_linear_knots_m=float(max_xyz),
                          maximum_orientation_deviation_from_slerp_knots_deg=float(np.rad2deg(max_angle)),
                          maximum_joint_deviation_from_linear_knots_rad=float(max_q_delta))
            stats.append(result)
            print(json.dumps(result),flush=True)
    return dict(hardware_ready=False, continuous_collision_certified=False, episodes=stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a = p.parse_args()
    if a.csv.exists() or a.report.exists():
        raise FileExistsError("Do not overwrite earlier checks")
    r = export(json.loads(a.plan.read_text()),a.urdf,a.csv)
    r.update(plan_sha256=sha256(a.plan),urdf_sha256=sha256(a.urdf),sample_csv_sha256=sha256(a.csv))
    with a.report.open("x") as f:
        json.dump(r,f,indent=2,allow_nan=False)
