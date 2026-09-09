#!/usr/bin/env python3
"""MuJoCo kinematic review images. Table XY extent shown is illustrative only."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import brentq
from replay_ik_trajectory_guarded import read_contact_ik

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['urdf','samples','scene','snapshot','output_dir']:p.add_argument('--'+n.replace('_','-'),dest=n,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(exist_ok=True,parents=True)
    scene=json.loads(a.scene.read_text());snap=json.loads(a.snapshot.read_text())
    seq=read_contact_ik(a.samples)
    tree=ET.parse(a.urdf)
    compiler=tree.find('./mujoco/compiler')
    compiler.set('strippath','false')
    m=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    m.vis.headlight.ambient[:]=.4;m.vis.headlight.diffuse[:]=.6
    arm=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'fr3_joint{i}') for i in range(1,8)]
    ai=m.jnt_qposadr[arm]
    grip=[i for i in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '').startswith('joint_gripper_')]
    gi=m.jnt_qposadr[grip];upper=float(np.min(m.jnt_range[grip,1]))
    left,right=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'link_gripper_contact_'+s) for s in ['left','right']]
    base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base')
    option=mujoco.MjvOption();option.geomgroup[:]=1;option.geomgroup[3]=0
    for i in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        m.geom_group[i]=3 if 'collision' in name else 0
    def opening(angle):
        d.qpos[gi]=angle;mujoco.mj_kinematics(m,d)
        return float(np.linalg.norm(d.xpos[left]-d.xpos[right]))
    poses=[('current_snapshot_width_assumed_max',np.asarray(snap['q']),.0887398049235344)]
    poses += [(f'episode_{ep}_first_frame',s[0]['q'],s[0]['width_m']) for ep,s in seq.items()]
    renderer=mujoco.Renderer(m,height=720,width=960)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',21)
    paths=[]
    for label,q,width in poses:
        output=a.output_dir/(label+'.jpg')
        if output.exists():raise FileExistsError(output)
        d.qpos[ai]=q;d.qpos[gi]=brentq(lambda v:opening(v)-width,0,upper,xtol=1e-12)
        mujoco.mj_forward(m,d)
        R=d.xmat[base].reshape(3,3);o=d.xpos[base]
        panels=[]
        for az in (90,145):
            cam=mujoco.MjvCamera();cam.lookat[:]=R@np.array([.3,0.,.36])+o
            cam.distance=1.65;cam.azimuth=az;cam.elevation=-18
            renderer.update_scene(d,camera=cam,scene_option=option)
            s=renderer.scene
            def box(half,position,rgba):
                mujoco.mjv_initGeom(s.geoms[s.ngeom],mujoco.mjtGeom.mjGEOM_BOX,np.array(half),R@np.array(position)+o,R.flatten(),np.array(rgba,dtype=np.float32));s.ngeom+=1
            # Visible slab only. Offline collision uses a larger conservative solid.
            box([.6,.65,.025],[scene['table_front_x_m']+.6,0.,scene['table_top_z_m']-.025],[.36,.43,.48,.75])
            box([.12,.14,.025],[-.04,0.,-.025],[.25,.28,.31,1.])
            panels.append(renderer.render().copy())
        im=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(im)
        draw.rectangle([0,0,1920,95],fill=(19,24,30))
        draw.text((18,10),label+' | same robot, two camera views',font=font,fill='white')
        draw.text((18,38),'Measured table front: base X = +133.6 mm | table top: base Z = +110 mm',font=font,fill='white')
        draw.text((18,66),'OFFLINE ONLY | displayed table width/depth illustrative | no hardware motion',font=font,fill=(255,210,110))
        im.save(output,quality=92);paths.append(str(output));print(output,flush=True)
    renderer.close()
    with (a.output_dir/'render_manifest.json').open('x') as f:json.dump(dict(images=paths,motion_commands_sent=0,table_front_x_m=scene['table_front_x_m'],table_top_z_m=scene['table_top_z_m'],rendered_table_extent_is_illustrative=True),f,indent=2)

if __name__=='__main__':main()
