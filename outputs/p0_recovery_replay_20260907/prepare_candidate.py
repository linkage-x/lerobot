"""Prepare isolated offline model/config. No robot control imports."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = Path('/home/nvidia/lerobot/outputs/p0_recovery_replay_20260907')
OLD = ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper_v2_p0.urdf'
BASE = ROOT / 'outputs/recalculation_20260907_193725/coordinate_chain_audit_20260907/p0_recovered_hk07_candidate.json'


def main():
    T = np.asarray(json.loads(BASE.read_text())['selected_T_world_base_candidate']['matrix_4x4'])
    model_dir = HERE / 'model'
    model_dir.mkdir(exist_ok=True)
    tree = ET.parse(OLD)
    root = tree.getroot()
    root.set('name', 'fr3_v2_p0_recovered_OFFLINE_CANDIDATE')
    joint = root.find("joint[@name='world_to_base']/origin")
    joint.set('xyz', ' '.join(map(str, T[:3, 3])))
    joint.set('rpy', ' '.join(map(str, Rotation.from_matrix(T[:3, :3]).as_euler('xyz'))))
    # Preserve all joint, inertia and geometry definitions; only relocate base
    # and make mesh paths unambiguous inside this separate output directory.
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((OLD.parent / mesh.attrib['filename']).resolve()))
    root.find('mujoco/compiler').set('meshdir', '/')
    dest = model_dir / 'fr3_v2_p0_recovered.urdf'
    with dest.open('xb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    cfg = yaml.safe_load((HERE / 'source/hikon_cube_tracking_offline/config_thor/april_cube_tracking_in_robot_base_thor.yaml').read_text())
    cfg['input']['dataset_root'] = str(REMOTE / 'dataset')
    cfg['calibration'].update(root_dir='/home/nvidia/lerobot/outputs/calibration',
                              intrinsics_run_name='thor_gmsl2_selfcal_0804_fisheye_intrinsics',
                              fixed_camera_run_name='calib_20260902_103833_extrinsics')
    snapshot = json.loads((BASE.parent / 'inputs/calibration_snapshot.json').read_text())
    cfg['calibration']['intrinsics_by_serial'] = {
        Path(item['path']).parent.name.split('_', 2)[2]: item['path']
        for item in snapshot['per_camera_intrinsics'].values()
    }
    cfg['ee_from_cube'].update(marker_to_tcp_calibration_path=str(REMOTE / 'source/hikon_cube_tracking_offline/config_thor/marker_to_tcp_calibration_20260825.json'),
                               target_urdf_path=str(REMOTE / 'closed_tcp_frame_hop.urdf'))
    cfg['contact_tcp']['enable'] = False  # supplied V2 formula is applied once downstream
    cfg['alignment'].update(method='none', first_frame_offset_enable=False)
    cfg['processing']['parallel_camera_workers'] = 7
    cfg['cube_tracker']['apriltag_detector']['nthreads'] = 1
    cfg['save_to_dataset'].update(sidecar_dir='derived/tracking_0902', write_parquet_inplace=False)
    cfg['output'].update(output_dir=str(REMOTE), run_name_mode='fixed', run_name='tracking', clean_run_dir=False)
    with (HERE / 'tracking_0902.yaml').open('x') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (HERE / 'preparation_manifest.json').open('x') as f:
        json.dump(dict(hardware_ready=False, base_matrix=T.tolist(), original_urdf=str(OLD),
                       original_urdf_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
                       model=str(dest), model_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                       base_candidate_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
                       tracker_commit='6b4f9e7aef458b71b318909b074646c5ea33545a',
                       production_modified=False), f, indent=2)
    print(dest)


if __name__ == '__main__':
    main()
