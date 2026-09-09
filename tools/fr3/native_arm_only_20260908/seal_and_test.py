"""Offline build verification only. Never creates a robot."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root=Path(__file__).resolve().parent
source=Path('/home/nvidia/box_api/replay_p0_once_20260908')
audit=Path('/home/nvidia/box_api/replay_p0_native_reuse_20260908')
sys.path.insert(0,str(root/'python'))
from panda_py import _core as core

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

checked=[]
for rel in ['src/controllers/joint_trajectory.cpp','src/controllers/joint_position.cpp',
            'src/motion/generators.cpp','src/motion/time_optimal/path.cpp',
            'src/motion/time_optimal/trajectory.cpp','src/panda.cpp']:
    assert sha(root/'native'/rel)==sha(source/'native'/rel),rel
    checked.append(rel)

q=np.array([0.,-.78,0.,-2.35,0.,1.57,.78])
for speed in [.01,.03]:
    traj=core.JointTrajectory([q.tolist(),(q+.01).tolist()],speed,.02,30.)
    ctrl=core.NativeJointTrajectoryController(traj,[300,300,300,300,120,80,30],
                                             [25,25,25,25,10,8,5],.001)
    assert traj.get_duration()>0 and ctrl.get_time()==0
    assert np.max(abs(np.asarray(traj.get_joint_positions(traj.get_duration()))-q-.01))<1e-9
    # shared ownership keeps trajectory alive after Python reference is dropped.
    del traj
    assert ctrl.get_time()==0

files=[root/'native_arm.py',root/'run_native_arm.sh',root/'seal_and_test.py']
files+=list((root/'python/panda_py').glob('*.so'))+list((root/'python/panda_py').glob('*.py'))
external=[audit/'benchmark_native.py']+list((audit/'build').glob('_native_plan_audit*.so'))
manifest=dict(files={str(p.relative_to(root)):sha(p) for p in files},
              external_files={str(p):sha(p) for p in external},
              unchanged_native_sources=checked,robot_instances=0,gripper_instances=0)
with (root/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('PASS: original native source equality, binding construction, endpoints, shared lifetime; no hardware instantiated')
