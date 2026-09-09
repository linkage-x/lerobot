"""Interactive local MuJoCo view of the recalculated IK; no hardware imports.

Run with mjpython on macOS. Space pauses, 0/1 selects episode, R restarts,
and left/right arrows step while paused. This is kinematic visualization.
"""
import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper_v2_p0.urdf'
OUT = ROOT / 'outputs/recalculation_20260907_193725'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed', type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.speed <= 4:
        parser.error('speed must be in (0,4]')
    with (OUT/'contact_ik.right.csv').open() as f:
        rows = list(csv.DictReader(f))
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [.65]*3
    model.vis.headlight.diffuse[:] = [.8]*3
    ids = [model.joint(f'fr3_joint{i}').id for i in range(1,8)]
    arm_indices = model.jnt_qposadr[ids]
    gripper_ids = [i for i in range(model.njnt) if 'gripper' in model.joint(i).name]
    gripper_indices = model.jnt_qposadr[gripper_ids]
    left = model.body('link_gripper_contact_left').id
    right = model.body('link_gripper_contact_right').id
    max_angle = min(model.jnt_range[gripper_ids,1])
    for i in range(model.ngeom):
        model.geom_group[i] = 3 if 'collision' in (model.geom(i).name or '') else 0
    episodes = {}
    def opening(angle):
        data.qpos[gripper_indices] = angle
        mujoco.mj_kinematics(model,data)
        return np.linalg.norm(data.xpos[left]-data.xpos[right])
    for ep in sorted({int(r['episode_index']) for r in rows}):
        frames=[]
        for row in rows:
            if int(row['episode_index']) != ep:
                continue
            data.qpos[arm_indices] = [float(row[f'fr3_joint{i}']) for i in range(1,8)]
            width = float(row['gripper_width_m'])
            angle = brentq(lambda a: opening(a)-width,0,max_angle,xtol=1e-12)
            data.qpos[gripper_indices] = angle
            frames.append((data.qpos.copy(),width,float(row['tcp_retreat_m'])))
        episodes[ep] = frames
    state = dict(episode=0,position=0.,paused=False)
    def on_key(key):
        if key==32:
            state['paused'] = not state['paused']
        elif key in (48,49):
            state['episode'] = key-48
            state['position'] = 0.
        elif key in (82,114):
            state['position'] = 0.
        elif key in (262,263):
            state['paused'] = True
            count=len(episodes[state['episode']])
            state['position']=float(np.clip(int(state['position'])+(1 if key==262 else -1),0,count-1))
    data.qpos[:] = episodes[0][0][0]
    mujoco.mj_forward(model,data)
    with mujoco.viewer.launch_passive(model,data,key_callback=on_key,
                                     show_left_ui=False,show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [.28,.02,.44]
            viewer.cam.distance=1.6
            viewer.cam.azimuth=145
            viewer.cam.elevation=-18
            viewer.opt.geomgroup[3]=0
        print('VIEWER_RUNNING: recalculated original IK; episodes 0/1; no robot connection',flush=True)
        previous=time.monotonic()
        while viewer.is_running():
            now=time.monotonic()
            elapsed=min(now-previous,.1)
            previous=now
            if not state['paused']:
                state['position'] += elapsed*60*args.speed
            seq=episodes[state['episode']]
            if state['position'] >= len(seq):
                state['episode']=1-state['episode']
                state['position']=0.
                seq=episodes[state['episode']]
            frame=min(int(state['position']),len(seq)-1)
            q,width,retreat=seq[frame]
            with viewer.lock():
                data.qpos[:]=q
                data.qvel[:]=0
                data.time=frame/60.
                mujoco.mj_forward(model,data)
                # Show the actual contact midpoint computed from both fingers.
                scene=viewer.user_scn
                scene.ngeom=0
                mujoco.mjv_initGeom(scene.geoms[0],mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.004]*3),(data.xpos[left]+data.xpos[right])/2,
                                   np.eye(3).ravel(),np.array([.1,1.,.3,1.],dtype=np.float32))
                scene.ngeom=1
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              'P0 / V2 contact IK\nEpisode / frame\nOpening / retreat\nPlayback\nKeys\nScope',
                              f'991 original IK targets (unchanged)\n{state["episode"]} / {frame+1} of {len(seq)}\n'
                              f'{width*1000:.2f} / {retreat*1000:.2f} mm\n'
                              f'{args.speed:g}x  {"PAUSED" if state["paused"] else "PLAYING"}\n'
                              'Space pause | 0/1 episode | R restart | arrows step\n'
                              'Kinematic only; no measured table; NOT hardware replay'))
            viewer.sync()
            time.sleep(.01)


if __name__=='__main__':
    main()
