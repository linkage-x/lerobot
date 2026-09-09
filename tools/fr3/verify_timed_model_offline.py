#!/usr/bin/env python3
"""Independent Pinocchio versus exported/native FK check, without devices."""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from panda_py import _core

from replay_ik_trajectory_guarded import JOINTS, sha256
from replay_timed_candidate import fk


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.plan.read_text())
    chain = plan["chain"]
    model = pin.buildModelFromUrdf(str(a.urdf))
    data = model.createData()
    q = pin.neutral(model)
    indices = [model.joints[model.getJointId(name)].idx_q for name in JOINTS]
    base = model.getFrameId("base")
    tcp = model.getFrameId("corenetic_gripper_ee")
    maximum = 0.
    count = 0
    for ep in plan["episodes"]:
        c = _core.TimedReplayCandidate(ep["times_s"],ep["coeff_descending_unit_interval"],
                                       chain["origins"],chain["axes"],chain["tail"])
        for t in np.unique(np.r_[ep["times_s"], np.linspace(0,c.duration(),1000)]):
            values = c.evaluate(float(t))
            q[indices] = values[:7]
            pin.framesForwardKinematics(model,data,q)
            independent = (data.oMf[base].inverse()*data.oMf[tcp]).homogeneous
            maximum = max(maximum,np.max(abs(independent-c.fk(values[:7]))),np.max(abs(independent-fk(chain,values[:7]))))
            count += 1
    assert maximum < 1e-10
    result = dict(hardware_ready=False, robot_instances_created=0,
                  pinocchio_version=pin.__version__, samples=count,
                  maximum_fk_matrix_element_error=float(maximum),
                  urdf_sha256=sha256(a.urdf),plan_sha256=sha256(a.plan),passed=True)
    with a.output.open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))
