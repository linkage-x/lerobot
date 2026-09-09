"""Kinematic MuJoCo playback of offline contact IK, no hardware access."""
import csv
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/recalculation_20260907_193725'
MODEL = ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper_v2_p0.urdf'


def main():
    rows = list(csv.DictReader((OUT/'contact_ik.right.csv').open()))
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    model.vis.headlight.ambient[:] = [.65,.65,.65]
    model.vis.headlight.diffuse[:] = [.8,.8,.8]
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,f'fr3_joint{i}') for i in range(1,8)]
    idx = model.jnt_qposadr[joints]
    grip = [i for i in range(model.njnt) if 'gripper' in mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i)]
    gi = model.jnt_qposadr[grip]
    left = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link_gripper_contact_left')
    right = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link_gripper_contact_right')
    def opening(q):
        data.qpos[gi] = q
        mujoco.mj_kinematics(model,data)
        return float(np.linalg.norm(data.xpos[left]-data.xpos[right]))
    upper = min(model.jnt_range[grip,1])
    renderer = mujoco.Renderer(model,height=720,width=960)
    cameras=[]
    for az in [145,55]:
        cam=mujoco.MjvCamera(); cam.lookat[:]=[.28,.02,.44];cam.distance=1.6;cam.azimuth=az;cam.elevation=-18
        cameras.append(cam)
    option=mujoco.MjvOption();option.geomgroup[:]=1
    # URDF visuals have no contact affinity; hide duplicate collision meshes.
    for i in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        model.geom_group[i]=3 if 'collision' in name else 0
    option.geomgroup[3]=0
    reports=[]
    for ep in [0,1]:
        seq=[r for r in rows if int(r['episode_index'])==ep]
        path=OUT/f'episode_{ep}_mujoco_2x_slow.mp4'
        width_errors=[];position_errors=[]
        with imageio.get_writer(path,fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
            for n,row in enumerate(seq):
                width=float(row['gripper_width_m'])
                data.qpos[idx]=[float(row[f'fr3_joint{i}']) for i in range(1,8)]
                q=brentq(lambda x:opening(x)-width,0,upper,xtol=1e-12)
                data.qpos[gi]=q
                mujoco.mj_forward(model,data)
                center=(data.xpos[left]+data.xpos[right])/2
                target=np.array([float(row[f'contact_target_{a}_m']) for a in 'xyz'])
                width_errors.append(abs(np.linalg.norm(data.xpos[left]-data.xpos[right])-width))
                position_errors.append(float(np.linalg.norm(center-target)))
                views=[]
                for cam in cameras:
                    renderer.update_scene(data,camera=cam,scene_option=option)
                    for pos,color in [(target,[1,.25,.05,1]),(center,[.1,1,.3,1])]:
                        s=renderer.scene
                        mujoco.mjv_initGeom(s.geoms[s.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.004]*3),pos,np.eye(3).flatten(),np.array(color,dtype=np.float32));s.ngeom+=1
                    views.append(renderer.render().copy())
                img=Image.fromarray(np.concatenate(views,axis=1));draw=ImageDraw.Draw(img)
                draw.rectangle((0,0,1920,61),fill=(20,24,31))
                draw.text((18,10),f'Episode {ep} | frame {n+1}/{len(seq)} | 0.5x speed | P0 + V2 | kinematic playback',fill='white')
                draw.text((18,32),f'Opening {width*1000:.2f} mm | retreat {float(row["tcp_retreat_m"])*1000:.2f} mm | red=target green=URDF contact center | no calibrated table geometry',fill='white')
                writer.append_data(np.asarray(img))
                if n==0: img.save(OUT/f'episode_{ep}_preview.jpg')
        report=dict(episode=ep,frames=len(seq),fps=30,playback_speed=.5,video=str(path),max_width_error_m=max(width_errors),max_urdf_contact_vs_formula_m=max(position_errors),kinematic=True,physics_tracking_simulated=False,environment_collision_checked=False)
        reports.append(report);print(json.dumps(report),flush=True)
    renderer.close()
    (OUT/'video_report.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
