"""Readable two-view MuJoCo review. No contact physics or robot control."""
import csv
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent


def main():
    if any(HERE.glob('episode_*_recovered_p0.mp4')):
        raise FileExistsError('Review video already exists')
    with (HERE / 'contact_ik.right.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert all(r['ik_ok'].lower() == 'true' for r in rows)
    m = mujoco.MjModel.from_xml_path(str(HERE / 'model/fr3_v2_p0_recovered.urdf'))
    m.vis.global_.offwidth, m.vis.global_.offheight = 960, 720
    m.vis.headlight.ambient[:] = [.3]*3
    m.vis.headlight.diffuse[:] = [.65]*3
    m.vis.headlight.specular[:] = [.05]*3
    d = mujoco.MjData(m)
    arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'fr3_joint{i}') for i in range(1,8)]
    ai = m.jnt_qposadr[arm]
    grip = [i for i in range(m.njnt) if 'gripper' in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,i)]
    gi = m.jnt_qposadr[grip]
    left, right = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'link_gripper_contact_'+side) for side in ['left','right']]
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base')
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        m.geom_group[i] = 3 if 'collision' in name else 0
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])) or ''
        if 'gripper' in body or body in ('base_link','link_sensor_ft'):
            m.geom_matid[i] = -1
            m.geom_rgba[i] = [.36,.48,.57,1.]
    options = mujoco.MjvOption()
    options.geomgroup[:] = 1
    options.geomgroup[3] = 0
    cams = []
    for az in [145,55]:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [.30,.02,.37]
        cam.distance, cam.azimuth, cam.elevation = 1.45, az, -16
        cams.append(cam)
    def opening(a):
        d.qpos[gi] = a
        mujoco.mj_kinematics(m,d)
        return float(np.linalg.norm(d.xpos[left]-d.xpos[right]))
    upper = float(np.min(m.jnt_range[grip,1]))
    renderer = mujoco.Renderer(m,height=720,width=960)
    report = []
    for ep in [0,1]:
        seq = [r for r in rows if int(r['episode_index'])==ep]
        path = HERE / f'episode_{ep}_recovered_p0.mp4'
        errors = []
        with imageio.get_writer(path,fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
            for k,row in enumerate(seq):
                d.qpos[ai] = [float(row[f'fr3_joint{i}']) for i in range(1,8)]
                width = float(row['gripper_width_m'])
                d.qpos[gi] = brentq(lambda a:opening(a)-width,0,upper,xtol=1e-12)
                mujoco.mj_forward(m,d)
                center = (d.xpos[left]+d.xpos[right])/2
                target = np.asarray([float(row[f'contact_target_{a}_m']) for a in 'xyz'])
                errors.append(float(np.linalg.norm(center-target)))
                height = float((d.xmat[base].reshape(3,3).T@(center-d.xpos[base]))[2]-.110)
                panels = []
                for cam in cams:
                    renderer.update_scene(d,camera=cam,scene_option=options)
                    scene = renderer.scene
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([.004]*3),target,np.eye(3).flatten(),np.array([1.,.35,.05,1.],dtype=np.float32))
                    scene.ngeom += 1
                    panels.append(renderer.render().copy())
                img = Image.fromarray(np.concatenate(panels,axis=1))
                draw = ImageDraw.Draw(img)
                draw.rectangle((0,0,1920,70),fill=(21,27,35))
                draw.text((18,10),f'Episode {ep} | {k+1}/{len(seq)} | corrected P0 candidate | 0.5x visual playback | two views of ONE robot',fill='white')
                draw.text((18,30),f'Gripper gap {width*1000:.2f} mm | retreat {float(row["tcp_retreat_m"])*1000:.2f} mm | contact center above +110 mm height: {height*1000:.1f} mm',fill='white')
                draw.text((18,50),'KINEMATIC REVIEW ONLY - no physical robot motion, no table footprint modeled, not a safety approval',fill=(255,198,92))
                writer.append_data(np.asarray(img))
                if k in (0,len(seq)//2):
                    img.save(HERE / f'episode_{ep}_review_{k:04d}.jpg')
        report.append(dict(episode=ep,frames=len(seq),video=str(path),fps=30,visual_speed=.5,
                           max_actual_contact_error_m=max(errors),hardware_ready=False,physics_tracking_simulated=False))
    renderer.close()
    with (HERE/'review_videos.json').open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':
    main()
